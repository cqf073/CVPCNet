"""
Query-Informed FSS
Extended from ADNet code by Hansen et al.
"""

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
from models.modules import MLP, Decoder, Supp_Decoder,MLPCompress,PrototypeFusion,FeatureExpandConv
from utils import *
from models.CLIPro import *

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3", alpha=0.9,dataset=None):# setting1 0.9  0.6
        super().__init__()
        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.alpha = torch.Tensor([alpha, 1 - alpha])
        self.alphaP = 0.7
        self.betaN = 0.3
        self.lam = 0.4
        self.temperature = 0.3
        if dataset == "CHAOST2" or dataset == "CMR":
            self.fg_num = 100
            self.bg_num = 600
            self.v = 256
        else:  # CT
            self.fg_num = 100
            self.bg_num = 600
            self.v = 257
        self.mlp1 = MLP(self.v, self.fg_num)
        self.mlp2 = MLP(self.v, self.bg_num)
        self.decoder1 = Decoder(self.fg_num)
        self.decoder2 = Decoder(self.bg_num)
        self.supp_decoder = Supp_Decoder()
        self.mlp3= MLPCompress(51,512)
        self.mlp4 = MLPCompress(301, 512)
        self.fuse = PrototypeFusion(2)
        self.pam_model = PAM(in_dim=512).to(self.device)
        self.cam_model = CAM(in_channels=512).to(self.device)  # Channel attention for 256 channels
        self.cosine_similarity_fn = F.cosine_similarity
        self.clip_model, self.preprocess = longclip.load("pretrained_model/hub/checkpoints/longclip-B.pt", device=self.device)
        self.expandSize = FeatureExpandConv(512, 512)
        self.avgTO2= nn.AdaptiveAvgPool2d(1)
    def forward(self, supp_imgs, supp_mask, qry_imgs, text_tokens,train=False, n_iters=0):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors 1 3 256 256
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images   1 3 256 256
                N x [B x 3 x H x W], list of tensors
        """
        text_tokens = text_tokens.cuda()
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1
        # mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # list = 2  1 1 1 512 64 64# B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'  list = 2 ,1 1 512 64 64
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        ###### Get threshold #######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        loss_PCMR = torch.zeros(1).to(self.device)
        loss_aux = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        MLN_loss = torch.zeros(1).to(self.device)
        outputs = []
        outputs_ = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                         range(len(supp_fts))]  # list list list tensor
            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]  # list list tensor

            supp_fts__ = [[self.getFeatures(supp_fts[0][[epi], way, shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]
            fg_prototypes_t = self.getPrototype(supp_fts__)

            input_image = supp_fts__[0][0].unsqueeze(-1).unsqueeze(-1)
            input_image = self.expandSize(input_image)
            output_feature_map = DSF(input_image, text_tokens, self.clip_model, self.pam_model, self.cam_model, self.cosine_similarity_fn)
            image_text_fts = [[self.avgTO2(output_feature_map).squeeze(-1).squeeze(-1)]]
            fg_prototypes_image_text = self.getPrototype(image_text_fts)#
            fg_prototypes_image_text = [fg_prototypes_image_text[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(0),fg_prototypes_image_text[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(0)]

            supp_pred = F.interpolate(supp_pred, size=img_size, mode='bilinear', align_corners=True)
            supp_pred = torch.cat((1.0 - supp_pred, supp_pred), dim=1)
            self_pred = supp_pred
            supp_pred = torch.argmax(supp_pred, dim=1, keepdim=True).squeeze(1) # 1 256 256

            fg_pts = [[self.get_fg_pts(supp_fts[0][[epi], way, shot], supp_mask[[epi], way, shot], supp_pred, self.v)
                       for shot in range(self.n_shots)] for way in range(self.n_ways)]

            fg_pts = self.get_all_prototypes(fg_pts)
            fg_pts_cl = self.mlp3(fg_pts[0])

            bg_pts = [[self.get_bg_pts(supp_fts[0][[epi], way, shot], supp_mask[[epi], way, shot], supp_pred, self.v)
                       for shot in range(self.n_shots)] for way in range(self.n_ways)]
            bg_pts = self.get_all_prototypes(bg_pts) #
            bg_pts_cl = self.mlp4(bg_pts[0])
            fg_sim = torch.stack(# 1 1 64 64
                [self.get_fg_sim(qry_fts[0][epi], fg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)
            bg_sim = torch.stack(# 1 1 64 64
                [self.get_bg_sim(qry_fts[0][epi], bg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)

            fg_pred = F.interpolate(fg_sim, size=img_size, mode='bilinear', align_corners=True)# 1 1 256 256
            bg_pred = F.interpolate(bg_sim, size=img_size, mode='bilinear', align_corners=True)# 1 1 256 256
            predscow = torch.cat([bg_pred, fg_pred], dim=1)
            predscow = torch.softmax(predscow, dim=1) #([1, 2, 256, 256])
            fp_l = []
            for n in range(len(supp_fts)):
                fp = torch.stack(fg_prototypes[n], dim=0)
                fp_l.append(fp.unsqueeze(-1).unsqueeze(-1))

            # ###### Get query predictions ######
            qry_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'

            # ###### Prototype Refinement (only for test) ######
            fg_prototypes_ = []
            if (not train) and n_iters > 0:  # iteratively update prototypes
                fp_l = []
                for n in range(len(qry_fts)):
                    fg_prototypes_.append(
                        self.updatePrototype(qry_fts[n], fg_prototypes[n], qry_pred[n], n_iters, epi))
                    fp = fg_prototypes_[n]
                    fp_l.append(fp.unsqueeze(-1).unsqueeze(-1))

                qry_pred = [torch.stack(
                    [self.getPred(qry_fts[n][epi], fg_prototypes_[n][way], self.thresh_pred[way]) for way in
                     range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            fp_ls = []
            bp_ls = []
            for n in range(len(qry_fts)):
                AFP, ABP, LBP = self.AP(qry_fts[n][epi], qry_pred[n], epi)
                fp = fp_l[n] * 0.9 + AFP * 0.1
                bp = ABP * 0.2 + LBP * 0.8
                fp_ls.append(fp)
                bp_ls.append(bp)

            for n in range(len(fp_ls)):
                fp_ls[n] = fp_ls[n]+fg_prototypes_image_text[n]
            qry_pred_new_fg = [torch.stack(
                [self.getSim(qry_fts[n][epi], fp_ls[n][way], self.thresh_pred[way]) for way in
                 range(fp_ls[n].shape[0])], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            qry_pred_new_bg = [torch.stack(
                [self.getSim(qry_fts[n][epi], bp_ls[n][way], self.thresh_pred[way]) for way in
                 range(bp_ls[n].shape[0])], dim=1) for n in range(len(qry_fts))]  # N x 1 x H' x W'
            # ####### Combine predictions of different feature maps ######
            qry_pred_up = [F.interpolate(qry_pred_new_fg[n], size=img_size, mode='bilinear', align_corners=True)#list=2, 1 1 256 256
                           for n in range(len(qry_fts))]
            qry_pred_bg_up = [F.interpolate(qry_pred_new_bg[n], size=img_size, mode='bilinear', align_corners=True)#list=2, 1 1 256 256
                            for n in range(len(qry_fts))]
            pred = [self.alpha[n] * qry_pred_up[n] for n in range(len(qry_fts))]
            pred_bg = [self.alpha[n] * qry_pred_bg_up[n] for n in range(len(qry_fts))]
            preds = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)  #  N x Wa x H' x W'
            preds_bg = torch.sum(torch.stack(pred_bg, dim=0), dim=0) / torch.sum(self.alpha)
            preds = torch.cat((preds_bg, preds), dim=1)# N x (1 + Wa) x H x W
            preds = preds.softmax(1) # 1 2 256 256
            outputs.append(preds)
            qry_pred_up_ = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                            for n in range(len(qry_fts))]
            pred_ = [self.alpha[n] * qry_pred_up_[n] for n in range(len(qry_fts))]
            preds_ = torch.sum(torch.stack(pred_, dim=0), dim=0) / torch.sum(self.alpha)
            preds_ = torch.cat((1.0 - preds_, preds_), dim=1)  # N x (1 + Wa) x H x W
            outputs_.append(preds_)
            p2 = torch.mean(torch.cat([fp_ls[0][0],fp_ls[1][0]],dim=0,),dim=0,keepdim=True).unsqueeze(0).unsqueeze(0).squeeze(-1).squeeze(-1)# 1 512
            fake_supp_fg_proto = torch.cat([p2,fg_prototypes_t[0].unsqueeze(0).unsqueeze(0)],dim=1)
            fake_qry_fts = supp_fts[epi]
            qry_pred_cl = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)
            qry_pred_cl = torch.cat([qry_pred_cl,fg_pred],dim=1)

            fake_qry_prototypes = torch.stack(
                [torch.stack([self.getPrototypes(fake_qry_fts[0, 0], qry_pred_cl[way, [shot]])
                              for shot in range(self.n_shots+1)], dim=0) for way in range(self.n_ways)], dim=0)  # 1 2 1 512
            dgpm_fg_proto = torch.cat([t.squeeze(-1).squeeze(-1).unsqueeze(0) for t in fg_pts_cl], dim=1)

            prototypes_sim = F.cosine_similarity(fake_supp_fg_proto, dgpm_fg_proto, dim=3)
            contribution_factor = torch.stack([torch.stack([(1.0 / (torch.sum(prototypes_sim, dim=1) + 1e-5)) *
                                                            prototypes_sim[way, [shot]] for shot in range(self.n_shots+1)],
                                                           dim=0) for way in range(self.n_ways)], dim=0)
            fake_qry_seg = torch.sum(preds * contribution_factor, dim=1)
            fake_qry_segs = torch.stack((1.0 - fake_qry_seg, fake_qry_seg), dim=1)

            ###### Prototype alignment loss ######
            if train:
                align_loss_epi , b_loss_epi , _ ,loss_aux_epi ,ssp_loss_epi= self.alignDualLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi], epi,fg_pts,bg_pts,self_pred)
                # align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                #                                 [qry_fts[n][epi] for n in range(len(qry_fts))],
                #                                 preds, supp_mask[epi])
                fp_loss_epi = self.fgProtoLoss([fg_prototypes[n] for n in range(len(supp_fts))],
                                              [supp_fts[n][epi] for n in range(len(supp_fts))],
                                              supp_mask[epi])
                proto_loss_epi = self.protoLoss([fp_ls[n] for n in range(len(qry_fts))],
                                              [bp_ls[n] for n in range(len(qry_fts))],
                                              [supp_fts[n][epi] for n in range(len(supp_fts))],
                                              supp_mask[epi])

                b_loss +=  b_loss_epi
                loss_aux += loss_aux_epi
                ssp_loss += ssp_loss_epi
                align_loss_epi = 0.2 * align_loss_epi + 0.4 * proto_loss_epi + 0.4 * fp_loss_epi
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        output_ = torch.stack(outputs_, dim=1)  # N x B x (1 + Wa) x H x W
        output_ = output_.view(-1, *output_.shape[2:])
        output = 0.5 * output + 0.5 * output_
        return output, align_loss / supp_bs , output_ ,b_loss/supp_bs ,loss_PCMR/ supp_bs ,loss_aux/supp_bs ,ssp_loss/supp_bs,fake_qry_segs

    def updatePrototype(self, fts, prototype, pred, update_iters, epi):
        prototype_0 = torch.stack(prototype, dim=0)
        n_ways = len(prototype)
        prototype_ = Parameter(torch.stack(prototype, dim=0))
        optimizer = torch.optim.Adam([prototype_], lr=0.01)

        while update_iters > 0:
            with torch.enable_grad():
                pred_mask = torch.sum(pred, dim=-3)
                pred_mask = torch.stack((1.0 - pred_mask, pred_mask), dim=1).argmax(dim=1, keepdim=True)
                pred_mask = pred_mask.repeat([*fts.shape[1:-2], 1, 1])
                bg_fts = fts[epi] * (1 - pred_mask)
                fg_fts = torch.zeros_like(fts[epi])
                for way in range(n_ways):
                    fg_fts += prototype_[way].unsqueeze(-1).unsqueeze(-1).repeat(*pred.shape) \
                              * pred_mask[way][None, ...]
                new_fts = bg_fts + fg_fts
                fts_norm = torch.sigmoid((fts[epi] - fts[epi].min()) / (fts[epi].max() - fts[epi].min()))
                new_fts_norm = torch.sigmoid((new_fts - new_fts.min()) / (new_fts.max() - new_fts.min()))
                bce_loss = nn.BCELoss()
                loss = bce_loss(fts_norm, new_fts_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.stack([self.getPred(fts[epi], prototype_[way], self.thresh_pred[way])
                                for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
            update_iters += -1
        return prototype_

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        # print('thresh', thresh)
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return pred

    def getSim(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C x H x W
        """
        sim = F.cosine_similarity(fts, prototype, dim=1) * self.scaler
        # pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return sim
    # def sim(self, feature_q, fg_proto, bg_proto):
    #     similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
    #     similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
    #     out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 4.0
    #     out = out.softmax(1)
    #     return out

    def AP(self, feature_q, out, epi):
        channel = feature_q.shape[1]
        n_queries = feature_q.shape[0]
        n_ways = out.shape[1]
        pred_fg = out.view(n_queries, n_ways, -1)
        pred_bg = (1 - out).view(n_queries, 1, -1)
        fg_ls = []
        bg_ls = []
        bg_local_ls = []
        mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
        for n in range(n_queries):
            cur_feat = feature_q[n].view(channel, -1)
            f_h, f_w = feature_q[n].shape[-2:]
            fg_l = []
            bg_l = []
            bg_local_l = []

            for way in range(n_ways):
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi][way], 12).indices]
                if (pred_fg[epi][way] > 0.8).sum() > 0:
                    fg_feat = cur_feat[:, (pred_fg[epi][way] > 0.8)]  # .mean(-1)
                # elif (pred_fg[epi][way] > 0.7).sum() > 0:
                #         fg_feat = cur_feat[:, (pred_fg[epi][way] > 0.7)]  # .mean(-1)
                elif (pred_fg[epi][way] > 0.6).sum() > 0:
                        fg_feat = cur_feat[:, (pred_fg[epi][way] > 0.6)]  # .mean(-1)
                else:
                    # if (pred_fg[epi][way] > 0.6).sum() > 0:
                    #     fg_feat = cur_feat[:, (pred_fg[epi][way] > 0.6)]  # .mean(-1)
                    # else:
                    #     fg_feat = cur_feat[:, torch.topk(pred_fg[epi][way], 12).indices]  # .mean(-1)
                    fg_feat = cur_feat[:, torch.topk(pred_fg[epi][way], 12).indices]  # .mean(-1)
                fg_proto = fg_feat.mean(-1)
                fg_l.append(fg_proto)
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)  # 1024, N3
            cur_feat_norm_t = cur_feat_norm.t()  # N3, 1024


            for way in range(pred_bg.shape[1]):
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi][way], 12).indices]
                if (pred_bg[epi][way] > 0.7).sum() > 0:
                    bg_feat = cur_feat[:, (pred_bg[epi][way] > 0.7)]  # .mean(-1)
                # elif (pred_bg[epi][way] > 0.6).sum() > 0:
                #     bg_feat = cur_feat[:, (pred_bg[epi][way] > 0.6)]  # .mean(-1)
                elif (pred_bg[epi][way] > 0.5).sum() > 0:
                    bg_feat = cur_feat[:, (pred_bg[epi][way] > 0.5)]  # .mean(-1)
                else:
                    bg_feat = cur_feat[:, torch.topk(pred_bg[epi][way], 12).indices]  # .mean(-1)
                # else:
                #     if (pred_bg[epi][way] > 0.5).sum() > 0:
                #         bg_feat = cur_feat[:, (pred_bg[epi][way] > 0.5)]  # .mean(-1)
                #     else:
                #         bg_feat = cur_feat[:, torch.topk(pred_bg[epi][way], 12).indices]  # .mean(-1)
                #     # bg_feat = cur_feat[:, torch.topk(pred_bg[epi][way], 12).indices]  # .mean(-1)
                bg_proto = bg_feat.mean(-1)
                bg_l.append(bg_proto)
                bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)  # 1024, N2
                bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0  # N3, N2
                bg_sim = bg_sim.softmax(-1)
                bg_proto_local = torch.matmul(bg_sim, bg_feat.t())  # N3, 1024
                bg_proto_local = bg_proto_local.t().view(channel, f_h, f_w).unsqueeze(0)  # 1024, N3
                bg_local_l.append(bg_proto_local)

            fg_ls.append(torch.stack(fg_l, dim=0).unsqueeze(1))
            bg_ls.append(torch.stack(bg_l, dim=0).unsqueeze(1))
            # fg_local_ls.append(torch.stack(fg_local_l, dim=0))
            bg_local_ls.append(torch.stack(bg_local_l, dim=0))

        # global proto
        new_fg = torch.cat(fg_ls, 1).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 1).unsqueeze(-1).unsqueeze(-1)
        # local proto
        # new_fg_local = torch.cat(fg_local_ls, 1)
        new_bg_local = torch.cat(bg_local_ls, 1)
        return new_fg, new_bg, new_bg_local
    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'  (batch个s)
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C
        return masked_fts

    def getPrototypes(self, fts, mask):
        """
        Args:
            fts: input features, expect shape: 1 C h w
            mask: binary mask, expect shape: 1 H W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        prototypes = torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return prototypes

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts
        return fg_prototypes

    def alignDualLoss(self, supp_fts, qry_fts, pred, fore_mask, epi,sup_fg_pts, sup_bg_pts, self_pred):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        loss_aux = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        loss_PCMR = torch.zeros(1).to(self.device)

        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                fp_l = []
                for n in range(len(supp_fts)):
                    fp = torch.stack(fg_prototypes[n], dim=0)
                    fp_l.append(fp.unsqueeze(-1).unsqueeze(-1))

                qry_fts_ = [[self.getFeatures(qry_fts[0], pred_mask[way + 1])]]
                fg_prototypes_ = self.getPrototype(qry_fts_)
                qry_pred = self.getSelfPred(qry_fts[0], fg_prototypes_[0])
                qry_pred = F.interpolate(qry_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                qry_pred = torch.cat((1.0 - qry_pred, qry_pred), dim=1)
                qry_pred = torch.argmax(qry_pred, dim=1, keepdim=True).squeeze(1)

                fg_pts_ = [[self.get_fg_pts(qry_fts[0], pred_mask[way + 1], qry_pred, self.v)]]
                fg_pts_ = self.get_all_prototypes(fg_pts_)
                fg_pts_t = self.mlp3(fg_pts_[0])
                bg_pts_ = [[self.get_bg_pts(qry_fts[0], pred_mask[way + 1], qry_pred, self.v)]]
                bg_pts_ = self.get_all_prototypes(bg_pts_)
                bg_pts_t= self.mlp4(bg_pts_[0])

                loss_aux += self.get_aux_loss(sup_fg_pts[way], fg_pts_[way], sup_bg_pts[way], bg_pts_[way])

                # Get predictions
                supp_pred_t = self.get_fg_sim(supp_fts[0][way, [shot]], fg_pts_[way])
                bg_pred_ = self.get_bg_sim(supp_fts[0][way, [shot]], bg_pts_[way])
                supp_pred_t = F.interpolate(supp_pred_t, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                bg_pred_ = F.interpolate(bg_pred_, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # Combine predictions
                predscow_ = torch.cat([bg_pred_, supp_pred_t], dim=1)
                predscow_ = torch.softmax(predscow_, dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                ssp_log_prob = torch.log(torch.clamp(self_pred, eps, 1 - eps))
                ssp_loss += self.criterion(ssp_log_prob, supp_label[None, ...].long()) / n_shots / n_ways
                fp_ls = []
                bp_ls = []
                for n in range(len(qry_fts)):
                    AFP, ABP, LBP = self.AP(supp_fts[n][way, [shot]], supp_pred[n].unsqueeze(0), epi)
                    fp = fp_l[n] * 0.9 + AFP * 0.1
                    bp = ABP * 0.2 + LBP * 0.8
                    fp_ls.append(fp)
                    bp_ls.append(bp)

                supp_pred_new_fg = [torch.stack(
                    [self.getSim(supp_fts[n][way, [shot]], fp_ls[n][way], self.thresh_pred[way]) for way in
                     range(fp_ls[n].shape[0])], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
                supp_pred_new_bg = [torch.stack(
                    [self.getSim(supp_fts[n][way, [shot]], bp_ls[n][way], self.thresh_pred[way]) for way in
                     range(bp_ls[n].shape[0])], dim=1) for n in range(len(qry_fts))]  # N x 1 x H' x W'  way=1

                # ####### Combine predictions of different feature maps ######
                supp_pred_up = [F.interpolate(supp_pred_new_fg[n], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                               for n in range(len(qry_fts))]
                # qry_pred_up = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                #               for n in range(len(qry_fts))]
                supp_pred_bg_up = [F.interpolate(supp_pred_new_bg[n], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                                  for n in range(len(qry_fts))]
                pred = [self.alpha[n] * supp_pred_up[n] for n in range(len(qry_fts))]
                pred_bg = [self.alpha[n] * supp_pred_bg_up[n] for n in range(len(qry_fts))]
                preds = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)  # N x Wa x H' x W'
                preds_bg = torch.sum(torch.stack(pred_bg, dim=0), dim=0) / torch.sum(self.alpha)

                preds = torch.cat((preds_bg, preds), dim=1)  # N x (1 + Wa) x H x W
                pred_ups = preds.softmax(1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways
        return loss, b_loss, loss_PCMR, loss_aux, ssp_loss

    def fgProtoLoss(self, fg_prototypes, supp_fts, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                # supp_pred_bg = [self.getPred_(supp_fts[n][way, [shot]], bg_prototypes[n][way], self.thresh_pred[way])
                #              for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((1-preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss
    def protoLoss(self, fg, bg, supp_fts, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                # Get predictions
                supp_pred_fg = [self.getSim(supp_fts[n][way, [shot]], fg[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred_bg = [self.getSim(supp_fts[n][way, [shot]], bg[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred_fg = [F.interpolate(supp_pred_fg[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]
                supp_pred_bg = [F.interpolate(supp_pred_bg[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds_fg = [self.alpha[n] * supp_pred_fg[n] for n in range(len(supp_fts))]
                preds_fg = torch.sum(torch.stack(preds_fg, dim=0), dim=0) / torch.sum(self.alpha)
                preds_bg = [self.alpha[n] * supp_pred_bg[n] for n in range(len(supp_fts))]
                preds_bg = torch.sum(torch.stack(preds_bg, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((preds_bg, preds_fg), dim=1)
                pred_ups = pred_ups.softmax(1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss


    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def getSelfPred(self, supp_fts, supp_vec):
        """
        Args:
            supp_fts: 1 x 512 x 64 x 64
            supp_vec: 1 x 512
        """
        supp_vec = supp_vec[..., None, None].expand(-1, -1, supp_fts.shape[-2], supp_fts.shape[-1])
        supp_pred = torch.cat([supp_fts, supp_vec, supp_vec], dim=1)
        supp_pred = self.supp_decoder(supp_pred)
        return supp_pred


    def get_fg_pts(self, features, mask, pred_mask,v):
        """
        Args:
        features: (1, 512, 64, 64)
        mask: (1, 256, 256)
        pred_mask: (1, 256, 256)
        """
        features_trans = F.interpolate(features, size=mask.shape[-2:], mode='bilinear', align_corners=True)# 1 512 64 64

        ie_mask = mask.squeeze(0) - torch.tensor(cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device)
        ie_mask = ie_mask.unsqueeze(0)
        ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) \
                       / (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) \
                           / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        add_mask = (pred_mask.float() + mask).long()
        mask1 = torch.zeros_like(mask)
        mask2 = torch.zeros_like(mask)
        mask1[add_mask == 2] = 1
        mask2[add_mask == 1] = 1
        mask1[mask == 0] = 0
        mask2[mask == 0] = 0
        fg_fts = self.get_fg_fts(features_trans, mask)  #(1, 512, 64, 64)
        fg_prototypes = self.mlp1(fg_fts.view(512, v * v)).permute(1, 0) #100 512

        if torch.sum(mask2[mask2 == 1]) > 0:
            hard_fg = self.get_random_pts(features_trans, mask2, 50)
            k = random.sample(range(len(fg_prototypes)), 50)
            fg_prototypes = torch.cat([fg_prototypes[k], hard_fg], dim=0)
        #           bg_prototypes 100 512 , origin_prototype 1 512, oe_prototype 1 512
        fg_prototypes = torch.cat([fg_prototypes, origin_prototype, ie_prototype], dim=0)
        return fg_prototypes

    def get_bg_pts(self, features, mask, pred_mask,v):
        """
        Args:
            features: (1, 512, 64, 64)
            mask: (1, 256, 256)
            pred_mask: (1, 256, 256)
        """
        bg_mask = 1 - mask
        features_trans = F.interpolate(features, size=bg_mask.shape[-2:], mode='bilinear', align_corners=True)

        oe_mask = torch.tensor(cv2.dilate(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device) - mask.squeeze(0)
        oe_mask = oe_mask.unsqueeze(0)
        oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) \
                       / (oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        origin_prototype = torch.sum(features_trans * bg_mask[None, ...], dim=(-2, -1)) \
                           / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        add_mask = (pred_mask.float() + mask).long()
        mask1 = torch.zeros_like(mask)
        mask2 = torch.zeros_like(mask)
        mask1[add_mask == 0] = 1
        mask2[add_mask == 1] = 1
        mask1[bg_mask == 0] = 0
        mask2[bg_mask == 0] = 0
        bg_fts = self.get_fg_fts(features_trans, bg_mask)
        bg_prototypes = self.mlp2(bg_fts.view(512, v * v)).permute(1, 0)
        if torch.sum(mask2[mask2 == 1]) > 0:
            hard_bg = self.get_random_pts(features_trans, mask2, 100)
            k = random.sample(range(len(bg_prototypes)), 500)
            bg_prototypes = torch.cat([bg_prototypes[k], hard_bg], dim=0)
        bg_prototypes = torch.cat([bg_prototypes, origin_prototype, oe_prototype], dim=0)
        return bg_prototypes
    def get_fg_fts(self, fts, mask):
        """
        Args:
            fts: (1, 512, 256, 256)
            mask: (1, 256, 256)
        """
        _, c, h, w = fts.shape
        # select masked fg features
        fg_fts = fts * mask[None, ...]
        bg_fts = torch.ones_like(fts) * mask[None, ...]
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])
        select_pts = self.get_random_pts(fts, mask, n_pts)
        index = bg_fts == 0
        fg_fts[index] = select_pts.permute(1, 0).reshape(512*n_pts)
        return fg_fts

    def get_random_pts(self, features_trans, mask, n_prototype):
        """
        Args:
            features_trans: (1, 512, 256, 256)
            mask: (1, 256, 256)
            n_prototype: int
        """
        features_trans = features_trans.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        features_trans = features_trans[mask == 1]
        if len(features_trans) >= n_prototype:
            k = random.sample(range(len(features_trans)), n_prototype)
            prototypes = features_trans[k]
        else:
            if len(features_trans) == 0:
                prototypes = torch.zeros(n_prototype, 512).to(self.device)
            else:
                r = n_prototype // len(features_trans)
                k = random.sample(range(len(features_trans)), (n_prototype - len(features_trans)) % len(features_trans))
                prototypes = torch.cat([features_trans for _ in range(r)], dim=0)
                prototypes = torch.cat([features_trans[k], prototypes], dim=0)
        return prototypes

    def get_all_prototypes(self, fg_fts):
        """
        Args:
            fg_fts: way x shot x [all x 512]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]
        return prototypes


    def get_fg_sim(self, fts, prototypes):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototypes: (102, 512)
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        fg_sim = self.decoder1(fg_sim)
        return fg_sim

    def get_bg_sim(self, fts, prototypes):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototypes: (602, 512)
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        bg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        bg_sim = self.decoder2(bg_sim)
        return bg_sim

    def get_aux_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
        """
        Args:
            sup_fg_pts: (102, 512)
            qry_fg_pts: (102, 512)
            sup_bg_pts: (602, 512)
            qry_bg_pts: (602, 512)
        """
        d1 = torch.mean(sup_fg_pts, dim=0, keepdim=True)
        d2 = torch.mean(qry_fg_pts, dim=0, keepdim=True)
        b1 = torch.mean(sup_bg_pts, dim=0, keepdim=True)
        b2 = torch.mean(qry_bg_pts, dim=0, keepdim=True)

        d1 = F.normalize(d1, dim=-1)
        d2 = F.normalize(d2, dim=-1)
        b1 = F.normalize(b1, dim=-1)
        b2 = F.normalize(b2, dim=-1)

        fg_intra = torch.matmul(d1, d2.transpose(0, 1)).squeeze(0).squeeze(0)
        bg_intra = torch.matmul(b1, b2.transpose(0, 1)).squeeze(0).squeeze(0)
        intra_loss = 2 - fg_intra - bg_intra

        zero = torch.zeros(1).squeeze(0)
        sup_inter = torch.matmul(d1, b1.transpose(0, 1))
        qry_inter = torch.matmul(d2, b2.transpose(0, 1))
        inter_loss = torch.max(zero, torch.mean(sup_inter)) + torch.max(zero, torch.mean(qry_inter))
        return intra_loss + inter_loss
def compute_masks(p1, p2, alpha=0.7, beta=0.3):
    pos_mask = (p1 > alpha) & (p2 > alpha)
    diff_mask = (p1 - p2).abs() > beta
    return pos_mask, diff_mask

def safe_mean_features(Fq, mask, topk_fallback=2048):
    B, C, H, W = Fq.shape
    if mask.dtype != torch.bool:
        mask = mask.bool()
    F_flat = Fq.view(B, C, -1)                        # [B,C,HW]
    m_flat = mask.view(B, 1, -1)                      # [B,1,HW]
    num_pos = m_flat.sum(dim=-1).clamp(min=1)         # [B,1], 防止除0
    proto_sum = (F_flat * m_flat).sum(dim=-1)         # [B,C]
    proto_tmp = proto_sum / num_pos                   # [B,C]
    return proto_tmp

def ema_update(c_old, c_new_est, momentum=0.9):
    if c_old is None:
        return c_new_est
    return momentum * c_old + (1.0 - momentum) * c_new_est

def proto_guided_prob(Fq, c_proto, temperature=8.0):
    B, C, H, W = Fq.shape
    F_norm = F.normalize(Fq, dim=1)                   # [B,C,H,W]
    c_norm = F.normalize(c_proto, dim=1).view(B, C, 1, 1)  # [B,C,1,1]
    sim = (F_norm * c_norm).sum(dim=1, keepdim=True)  # [B,1,H,W], 范围约[-1,1]
    p_proto = torch.sigmoid(temperature * sim)        # [B,1,H,W]
    return p_proto

def reconcile_predictions(p1, p2, p_proto, diff_mask, lam=0.5):
    keep = (~diff_mask).float()
    edit = diff_mask.float()
    p1_corr = keep * p1 + edit * ((1 - lam) * p1 + lam * p_proto)
    p2_corr = keep * p2 + edit * ((1 - lam) * p2 + lam * p_proto)
    return p1_corr, p2_corr

def prototype_contrastive_loss(Fq, c_proto, pos_mask, diff_mask, tau=0.2):
    B, C, H, W = Fq.shape
    F_norm = F.normalize(Fq, dim=1)                   # [B,C,H,W]
    c = F.normalize(c_proto, dim=1).view(B, C, 1, 1)  # [B,C,1,1]
    sim = (F_norm * c).sum(dim=1, keepdim=True)       # [B,1,H,W]
    pos_w = pos_mask.float()
    neg_w = diff_mask.float()
    pos_den = pos_w.sum().clamp(min=1.0)
    neg_den = neg_w.sum().clamp(min=1.0)
    pos_loss = ((1.0 - sim) * pos_w).sum() / pos_den
    neg_loss = ((1.0 - sim) * neg_w).sum() / neg_den
    loss = (pos_loss + neg_loss) / tau
    return loss