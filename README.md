# Self-supervised-learning
## Requirements  
`pip install termcolor opencv-python tensorboard`

## Train and eval
#### The pre-training stage:
`output_dir="./output/imagenet/K65536"`    
`input_dir="./Imagenet"`
   `python -m torch.distributed.launch  --master_port 12347 --nproc_per_node=8 \ `
    `train.py\`
    `--data-dir ${data_dir} \`
    `--output-dir ${output_dir}`  
    Set --amp-opt-level to O1, O2, or O3 for mixed precision training. 
 #### The linear evaluation stage:
 `python -m torch.distributed.launch --nproc_per_node=4 \`
  ` --data-dir ${data_dir}`
  ` eval.py \`  
  ` --pretrained-model ${output_dir}/current.pth \`
  `  --output-dir ${output_dir}/eval`
