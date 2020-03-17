#!/bin/bash

data_dir="./imagenet"
output_dir="./output/moco_v1"
python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
    train_v1.py \
    --data-dir ${data_dir} \
    --amp-opt-level O2 \
    --output-dir ${output_dir}

python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    eval.py \
    --data-dir ${data_dir} \
    --pretrained-model ${output_dir}/current.pth \
    --output-dir ${output_dir}/eval



