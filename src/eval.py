import argparse
import re
import math
import json


def find_last_number(text):
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', re.sub(r',', '', text))
    return float(numbers[-1]) if numbers else None


def find_last_letter(text):
    target_letters = {'A', 'B', 'C', 'D', 'E'}
    for char in reversed(text):
        if char.upper() in target_letters:
            return char.upper()
    return "none"


def compare_numbers(a, b):
    try:
        return 1 if math.isclose(float(a), float(b), rel_tol=1e-9) else 0
    except:
        return 0


def compare_letters(a, b):
    return 1 if str(a).upper() == str(b).upper() else 0


def get_eval_config(dataset):
    configs = {
        'gsm8k': {
            'answer_key': 'answer',
            'extract_output': find_last_number,
            'extract_answer': lambda x: float(x),
            'compare': compare_numbers
        },
        'gsm-h': {
            'answer_key': 'answer',
            'extract_output': find_last_number,
            'extract_answer': lambda x: float(x),
            'compare': compare_numbers
        },
        'svamp': {
            'answer_key': 'answer',
            'extract_output': find_last_number,
            'extract_answer': lambda x: float(x),
            'compare': compare_numbers
        },
        'mathqa': {
            'answer_key': 'answers',
            'extract_output': find_last_letter,
            'extract_answer': lambda x: str(x).strip().upper(),
            'compare': compare_letters
        },
        'math23k': {
            'answer_key': 'answer',
            'extract_output': find_last_number,
            'extract_answer': lambda x: float(re.search(r'\d+\.?\d*', x).group()),
            'compare': compare_numbers
        }
    }
    return configs.get(dataset)


def main():
    parser = argparse.ArgumentParser(description='评估不同数据集的模型输出')
    parser.add_argument('--input', required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', required=True, help='输出JSONL文件路径')
    parser.add_argument('--dataset', required=True,
                        choices=['gsm8k', 'gsm-h', 'svamp', 'mathqa', 'math23k'],
                        help='数据集名称')

    args = parser.parse_args()
    config = get_eval_config(args.dataset)

    if not config:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    correct = 0
    total = 0

    with open(args.input, 'r', encoding='utf-8') as infile, \
            open(args.output, 'w', encoding='utf-8') as outfile:

        for line in infile:
            data = json.loads(line)
            output = data.get('output', '')
            answer = data.get(config['answer_key'], '')

            pred = config['extract_output'](output)
            gt = config['extract_answer'](answer)

            label = config['compare'](pred, gt) if pred and gt else 0
            data['label'] = label

            correct += label
            total += 1

            outfile.write(json.dumps(data) + '\n')

    accuracy = 100 * correct / total if total else 0
    print(f"评估完成 | 数据集: {args.dataset}")
    print(f"正确率: {accuracy:.2f}% ({correct}/{total})")
    print(f"结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
