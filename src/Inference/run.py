import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import json
from tqdm import tqdm
import argparse

# Set OpenAI's API key and API base to use vLLM's API server.
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def split_list(input_list, n):
    """
    Split a list into multiple sublists, each containing n elements.
    """
    return [input_list[i:i + n] for i in range(0, len(input_list), n)]


def choose_prompt(dataset_name, question, option=None, stage=None):
    """
    Choose the appropriate prompt based on the dataset name and stage.
    """
    if stage == "EULER":
        # For EULER stage, the prompt is directly read from the input file
        return question  # Return the question as the prompt
    elif dataset_name in ["gsm8k", "gsm-h", "svamp"]:  # First category
        return f'Solve the problems step by step and put your final answer within \\boxed{{}}. **Problems**: {question}'
    elif dataset_name == "mathqa":
        # For mathqa, include both the question and options
        return f"Please solve the following problem: {question} Options: {option}"
    elif dataset_name == "math23k":
        return f"逐步解决下面问题，并将您的最终答案放在 \\boxed{{}} 中。 **问题**： {question}"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params, dataset_name, stage):
    data = []
    with jsonlines.open(json_filename) as infile:
        for item in infile:
            if stage == "EULER":
                # For EULER stage, read the prompt directly from the input file
                prompt = item.get('question')  # Ensure the key for prompt is correct
                # question = prompt  # Use the prompt as the question
                question = None
                option = None
                answer = item.get('answer')  # Ensure the key for answer is correct
            else:
                # For other stages, use the existing logic
                question = item.get('Problem')  # Ensure correct key for question
                option = item.get('options')  # Ensure the key for option is correct
                prompt = choose_prompt(dataset_name, question, option, stage)  # Generate prompt
                answer = item.get('correct')  # Ensure the key for answer is correct

            group = {'question': question, 'option': option, 'answer': answer, 'prompt': prompt}
            data.append(group)

    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt['prompt']}], tokenize=False, add_generation_prompt=True) for prompt in data]
    split_texts = split_list(texts, batch_size)

    if stage == "test":
        # Generate only one answer
        sampling_params = SamplingParams(n=1, temperature=0.8, top_p=0.9, repetition_penalty=1.05, max_tokens=1024)
    elif stage == "data_collection":
        # Generate five answers
        sampling_params = SamplingParams(n=5, temperature=0.8, top_p=0.9, repetition_penalty=1.05, max_tokens=1024)
    elif stage == "EULER":
        # Generate one answer for EULER stage
        sampling_params = SamplingParams(n=1, temperature=0.8, top_p=0.9, repetition_penalty=1.05, max_tokens=1024)
    else:
        raise ValueError(f"Invalid stage: {stage}")

    for item in tqdm(split_texts):
        outputs = llm.generate(item, sampling_params)
        if stage == "test":
            for i, output in enumerate(outputs):
                result = output.outputs[0].text
                data[i]['output'] = result
        elif stage == "data_collection":
            for i, output in enumerate(outputs):
                data[i].update({
                    "output0": output.outputs[0].text,
                    "output1": output.outputs[1].text,
                    "output2": output.outputs[2].text,
                    "output3": output.outputs[3].text,
                    "output4": output.outputs[4].text,
                })
        elif stage == "EULER":
            for i, output in enumerate(outputs):
                result = output.outputs[0].text
                data[i]['output'] = result

    with open(file_name, "w", encoding="utf-8") as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + "\n")
    print(f"Data has been successfully written to {file_name}")


def main():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument('--json_filename', type=str, required=True, help="Path to input JSONL file")
    parser.add_argument('--file_name', type=str, required=True, help="Path to output JSONL file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the tokenizer")
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=["gsm8k", "gsm-h", "svamp", "mathqa", "math23k"], help="Dataset name")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for inference")
    parser.add_argument('--gpu', type=str, default="6", help="CUDA GPU device to use")
    parser.add_argument('--stage', type=str, required=True, choices=["test", "data_collection", "EULER"],
                        help="Stage: test, data_collection, or EULER")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    sampling_params = SamplingParams(n=1, temperature=0.8, top_p=0.9, repetition_penalty=1.05, max_tokens=1024)

    llm = LLM(model=args.model_path, gpu_memory_utilization=0.9, max_model_len=4096, trust_remote_code=True,
              tensor_parallel_size=1)

    process_data(args.json_filename, args.file_name, llm, args.batch_size, tokenizer, sampling_params,
                 args.dataset_name, args.stage)


if __name__ == "__main__":
    main()
