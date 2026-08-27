#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='SABS'
NWORKER=0
RUNS=1
ALL_EV=(4 3 2 1 0) # 2-fold cross validation (0, 1, 2, 3, 4)
#ALL_EV=(2 3)
#TEST_LABEL=[1,2,3,6]
TEST_LABEL=[1,6]
#TEST_LABEL=[2,3]
#EXCLUDE_LABEL=None
EXCLUDE_LABEL=[2,3]
#EXCLUDE_LABEL=[1,6]
USE_GT=False
###### Training configs ######
NSTEP=60000
DECAY=0.98

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=60000 # interval for saving snapshot
SEED=2021

echo ========================================================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="train_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./exps_on_${DATASET}_FSS"

  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi

  python3 train.py with \
  mode='train' \
  dataset=$DATASET \
  num_workers=$NWORKER \
  n_steps=$NSTEP \
  eval_fold=$EVAL_FOLD \
  test_label=$TEST_LABEL \
  exclude_label=$EXCLUDE_LABEL \
  use_gt=$USE_GT \
  max_iters_per_load=$MAX_ITER \
  seed=$SEED \
  save_snapshot_every=$SNAPSHOT_INTERVAL \
  lr_step_gamma=$DECAY \
  path.log_dir=$LOGDIR
done