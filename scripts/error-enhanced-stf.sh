export CUDA_VISIBLE_DEVICES="you gpu device id"

# 运行训练命令
llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "your model path" \
  --dataset "STF-DATASET" \
  --dataset_dir ./data \
  --template baichuan2 \
  --finetuning_type lora \
  --output_dir "your output path" \
  --overwrite_output_dir \
  --cutoff_len 2048 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --warmup_steps 20 \
  --save_steps 100 \
  --eval_steps 25 \
  --evaluation_strategy steps \
  --load_best_model_at_end \
  --learning_rate 5e-5 \
  --num_train_epochs 10.0 \
  --val_size 0.1 \
  --plot_loss \
  --fp16 \
  --report_to tensorboard
