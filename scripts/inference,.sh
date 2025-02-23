python ./src/inference/run.py \
    --json_filename "The path to your input jsonl file" \ 
    --file_name "The path to your output jsonl file" \
    --model_path "your model path" \
    --tokenizer_path "your model path" \
    --dataset_name "choose dataset from ["gsm8k", "gsm-h", "svamp", "mathqa", "math23k"] \
    --gpu "your gpu device id" \
    --stage EULER
