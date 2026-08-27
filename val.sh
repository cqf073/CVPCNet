#!/bin/bash
# test a model to segment abdominal/cardiac MRI
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='CHAOST2'
#DATASET='CMR'
#DATASET='SABS'
NWORKER=16
RUNS=1
ALL_EV=(2)  # 2-fold cross validation (0, 1, 2, 3, 4)
#ALL_EV=(4 0 1 2 3)
TEST_LABEL=[1,2,3,4]
#TEST_LABEL=[1,2,3,6]
#TEST_LABEL=[1,2,3]
#TEST_LABEL=[2,3]
#TEST_LABEL=[1,6]
#TEST_LABEL=[1,4]

###### Training configs ######
NSTEP=60000
DECAY=0.98

MAX_ITER=1000 # defines the size ofan epoch
SNAPSHOT_INTERVAL=60000 # interval for saving snapshot
SEED=2021

N_PART=3 # defines the number of chunks for evaluation
ALL_SUPP=(0 1 2 3 4) # CHAOST2: 0-4, CMR: 0-7 SABS:0-6
echo $ALL_SUPP
echo =======================================================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="test_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./results"

  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi
  for SUPP_IDX in "${ALL_SUPP[@]}"
  do
    echo "Current SUPP_IDX: $SUPP_IDX"
    echo "Current EVAL_FOLD: $EVAL_FOLD"
    RELOAD_MODEL_PATH="./exps_on_CHAOST2_FSS/MPANet_train_CHAOST2_cv2/13/snapshots/60000.pth"  #CHAOS
    python test.py with \
    mode="test" \
    dataset=$DATASET \
    num_workers=$NWORKER \
    n_steps=$NSTEP \
    eval_fold=$EVAL_FOLD \
    max_iters_per_load=$MAX_ITER \
    supp_idx=$SUPP_IDX \
    test_label=$TEST_LABEL \
    seed=$SEED \
    n_part=$N_PART \
    reload_model_path=$RELOAD_MODEL_PATH \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR
    done
done