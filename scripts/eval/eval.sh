#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1
# MASTER_ADDR=localhost
# MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${2-2}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK"

# model
BASE_PATH=${1-"/hpc2hdd/home/bhuangas/jhaidata/ESO"}
CKPT_NAME="gpt2-medium"

CKPT=""

DATA_DIR=""

# hp
BATCH_SIZE=4
LR=0.0002
GRAD_ACC=1
EVAL_BATCH_SIZE=16
# length
MAX_LENGTH=512

SAVE_PATH=""

# seed
SEED=10

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 5e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 5"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 512"
# runtime
OPTS+=" --do-eval"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# OPTS+=" --deepspeed_config /hpc2hdd/home/bhuangas/jhaidata/text_sum/configs/ds_config_fix.json"
# type
OPTS+=" --model-type opt"
OPTS+=" --type eval_main"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0" # will be overided by top_k_list
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
OPTS+=" --max-new-tokens 64"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="deepspeed --master_port 29600 ${BASE_PATH}/eval_main.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
