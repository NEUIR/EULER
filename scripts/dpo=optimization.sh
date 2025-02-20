export CUDA_VISIBLE_DEVICES="your gpu device id"

#!/bin/bash

python src/dpo-optimization.py \
  --model_name_or_path "your model path" \
  --train_data_path "your train dataset path" \
  --eval_data_path "your dev dataset path" \
  --cache_dir "your cache path" \
  --output_dir "your output path" \
  --learning_rate 5e-05 \
  --max_length 1024 \
  --do_train True \
  --do_eval True \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --use_lora True \
  --logging_dir "your log path" \
  --log_level "info" \
  --save_strategy "steps" \
  --save_steps 100 \
  --overwrite_output_dir True \
  --neftune_noise_alpha 5 \
  --num_train_epochs 10 
