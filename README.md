# EULER: Enhancing the Reasoning Ability of Large Language Models through Error-Induced Learning

This repository contains the source code for the paper: [EULER: Enhancing the Reasoning Ability of Large Language Models through Error-Induced Learning]().

[![GitHub](https://img.shields.io/badge/GitHub-EULER-black?logo=github)](https://github.com/NEUIR/EULER/edit/main/README.md)
[![arXiv](https://img.shields.io/badge/arXiv-anonymous-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/xxxx.xxxxx)
[![HuggingFace-Paper](https://img.shields.io/badge/HuggingFace-anonymous-yellow?logo=huggingface)](https://huggingface.co/papers/xxxx.xxxxx)
[![HuggingFace-EULER](https://img.shields.io/badge/HuggingFace-EULER-yellowgreen)](https://huggingface.co/qizheyanger/EULER)

<div align="center">
<p align="center" dir="auto">

• 🎯 [Overview](#overview) 
• ⚙️ [Requirements](#requirements)
• 🔧 [Reproduction Guide](#reproduction-guide)
</p>
<p align="center" dir="auto">

• ✈️ [Experimental Result](#experimental-result) 
• 📃 [Acknowledgement](#acknowledgement) 
• 📝 [Citation](#citation)
• 📨 [Contact](#contact)
</p>
</div>


## Overview

EULER is an innovative framework designed to enhance the mathematical reasoning capabilities of Large Language Models (LLMs) by strategically optimizing the generation and utilization of solution errors. At its core, EULER introduces a novel error exposure model that increases the likelihood of generating self-made solution errors while leveraging high-quality solutions from a superior LLM to regularize the overall generation quality.


![](figs/图片1.png)


## Requirements

### 1. Python Environment:

Install the following packages using Pip or Conda under this environment.

```
python >= 3.8
torch == 1.12.1
recbole == 1.2.0
datasets == 3.1.0
transformers == 4.22.2
sentencepiece == 0.2.0
faiss-cpu == 1.8.0.post1
scikit-learn >= 1.1.2
numpy >= 1.17.2
pandas >= 1.0.0
tqdm
jsonlines
networkx
```
### 2. Install LLaMA-Factory.
Refer to [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for detailed instructions.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Data-Collection

```bash
bash scripts/Data-Collection.sh 
```

## Train

### 1. Dpo-Optimization

```bash
bash scripts/dpo.sh
```

### 2. Error-Enhanced-SFT

```bash
bash scripts/error-enhanced-sft.sh 
```

## Inference

```bash
bash scripts/inference.sh 
```

## Eval
```bash
bash scripts/eval.sh 
```
