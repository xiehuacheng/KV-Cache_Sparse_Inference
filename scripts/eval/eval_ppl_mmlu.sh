#!/bin/bash

MODEL_TYPE=${1}     # vicuna-v1.5
MODEL_SIZE=${2}     # 7b
QUANT_METHOD=${3}   # awq
BITS=${4}           # 3
KV_BITS=${5}           # 4
TASK=${6}           # ppl/mmlu/qa/streaming
BATCH_SIZE=${7}     # 1
PORT=${8}           # 29500
DEVICE=${9}         # 0
START_SIZE=${10}     # 8
RECENT_SIZE=${11}   # 1016
K=${12}            # 512
HH_SIZE=${13}      # 512
HH_RECENT_SIZE=${14}    # 512

set -v

if [ ${TASK} == "ppl" ];
then
    # vanilla
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --tasks ${TASK} --batch_size ${BATCH_SIZE}

    # streamingllm
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --tasks ${TASK}  --batch_size ${BATCH_SIZE}\
        --enable_streaming_pos_shift --start_size ${START_SIZE} --recent_size ${RECENT_SIZE}

    # h2o
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --tasks ${TASK}  --batch_size ${BATCH_SIZE}\
        --enable_h2o --k ${K} --heavy_hitter_size ${HH_SIZE} --heavy_hitter_recent_size ${HH_RECENT_SIZE}

    # streamingllm + h2o
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --tasks ${TASK}  --batch_size ${BATCH_SIZE}\
        --enable_streaming_pos_shift --start_size ${START_SIZE} --recent_size ${RECENT_SIZE} \
        --enable_h2o --k ${K} --heavy_hitter_size ${HH_SIZE} --heavy_hitter_recent_size ${HH_RECENT_SIZE}

    if [ "${QUANT_METHOD}" != "fp32" ] && [ "${QUANT_METHOD}" != "fp16" ];
    then
        # intactkv
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python ./intactkv_eval.py \
            --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
            --quant_method ${QUANT_METHOD} \
            --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
            --intactkv \
            --tasks ${TASK} --batch_size ${BATCH_SIZE}

        # streamingllm + intactkv
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python ./intactkv_eval.py \
            --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
            --quant_method ${QUANT_METHOD} \
            --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
            --intactkv \
            --tasks ${TASK} --batch_size ${BATCH_SIZE} \
            --enable_streaming_pos_shift --start_size ${START_SIZE} --recent_size ${RECENT_SIZE}
        
        # h2o + intactkv
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python ./intactkv_eval.py \
            --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
            --quant_method ${QUANT_METHOD} \
            --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
            --intactkv \
            --tasks ${TASK} --batch_size ${BATCH_SIZE} \
            --enable_h2o --k ${K} --heavy_hitter_size ${HH_SIZE} --heavy_hitter_recent_size ${HH_RECENT_SIZE}

        # streamingllm + h2o + intactkv
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python ./intactkv_eval.py \
            --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
            --quant_method ${QUANT_METHOD} \
            --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
            --intactkv \
            --tasks ${TASK} --batch_size ${BATCH_SIZE} \
            --enable_streaming_pos_shift --start_size ${START_SIZE} --recent_size ${RECENT_SIZE} \
            --enable_h2o --k ${K} --heavy_hitter_size ${HH_SIZE} --heavy_hitter_recent_size ${HH_RECENT_SIZE}
    fi
else
    # vanilla
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --tasks ${TASK} --batch_size ${BATCH_SIZE}

    # # streamingllm
    # CUDA_VISIBLE_DEVICES=${DEVICE} \
    # python ./intactkv_eval.py \
    #     --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
    #     --quant_method ${QUANT_METHOD} \
    #     --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
    #     --tasks ${TASK} --batch_size ${BATCH_SIZE} \
    #     --enable_streaming_pos_shift --start_size ${START_SIZE} --recent_size ${RECENT_SIZE}

    # # h2o
    # CUDA_VISIBLE_DEVICES=${DEVICE} \
    # python ./intactkv_eval.py \
    #     --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
    #     --quant_method ${QUANT_METHOD} \
    #     --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
    #     --tasks ${TASK} --batch_size ${BATCH_SIZE} \
    #     --enable_h2o --k ${K} --heavy_hitter_size ${HH_SIZE} --heavy_hitter_recent_size ${HH_RECENT_SIZE}

    # # streamingllm + h2o
    # CUDA_VISIBLE_DEVICES=${DEVICE} \
    # python ./intactkv_eval.py \
    #     --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
    #     --quant_method ${QUANT_METHOD} \
    #     --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
    #     --tasks ${TASK} --batch_size ${BATCH_SIZE} \
    #     --enable_streaming_pos_shift --start_size ${START_SIZE} --recent_size ${RECENT_SIZE} \
    #     --enable_h2o --k ${K} --heavy_hitter_size ${HH_SIZE} --heavy_hitter_recent_size ${HH_RECENT_SIZE}

    if [ "${QUANT_METHOD}" != "fp32" ] && [ "${QUANT_METHOD}" != "fp16" ];
    then
        # intactkv
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python ./intactkv_eval.py \
            --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
            --quant_method ${QUANT_METHOD} \
            --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
            --intactkv \
            --tasks ${TASK} --batch_size ${BATCH_SIZE}

        # streamingllm + intactkv
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python ./intactkv_eval.py \
            --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
            --quant_method ${QUANT_METHOD} \
            --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
            --intactkv \
            --tasks ${TASK} --batch_size ${BATCH_SIZE} \
            --enable_streaming_pos_shift --start_size ${START_SIZE} --recent_size ${RECENT_SIZE}
        
        # h2o + intactkv
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python ./intactkv_eval.py \
            --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
            --quant_method ${QUANT_METHOD} \
            --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
            --intactkv \
            --tasks ${TASK} --batch_size ${BATCH_SIZE} \
            --enable_h2o --k ${K} --heavy_hitter_size ${HH_SIZE} --heavy_hitter_recent_size ${HH_RECENT_SIZE}

        # streamingllm + h2o + intactkv
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python ./intactkv_eval.py \
            --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
            --quant_method ${QUANT_METHOD} \
            --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
            --intactkv \
            --tasks ${TASK} --batch_size ${BATCH_SIZE} \
            --enable_streaming_pos_shift --start_size ${START_SIZE} --recent_size ${RECENT_SIZE} \
            --enable_h2o --k ${K} --heavy_hitter_size ${HH_SIZE} --heavy_hitter_recent_size ${HH_RECENT_SIZE}
    fi
fi
