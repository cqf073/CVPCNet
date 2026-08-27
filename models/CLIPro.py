import torch
import torch.nn as nn
import torch.nn.functional as F
class CAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
            avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
            return self.sigmoid(out) + x

class PAM(nn.Module):
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.channel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

def DSF(input_image, text_tokens, clip_model, pam_model, cam_model, cosine_similarity_fn):
    device = next(clip_model.parameters()).device
    input_image = input_image.to(device)
    text_tokens = text_tokens.to(device)
    text_embedding = text_tokens
    T = text_embedding.float()
    F_1 = input_image
    text_weight = T.unsqueeze(-1).unsqueeze(-1)
    text_weight = torch.sigmoid(text_weight)
    F_fusion = F_1 * text_weight
    F_fusion = F_fusion.view(F_1.shape)
    F_attention = cam_model(F_fusion)
    T_S = T.view(1, 512, 1, 1).expand(1, 512, 64, 64)
    F_S = F_attention.view(F_1.shape)
    similarity = cosine_similarity_fn(T_S, F_S)
    refined_F = pam_model(F_attention)
    F_final = refined_F + F_attention
    return F_final