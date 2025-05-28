# Hallucination Detection vs Fact Verification

This repository provides implementations for hallucination detection and fact verification methods for large language models (LLMs). The project includes multiple detection approaches and evaluation frameworks across various QA datasets.

## Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Hallucination Detection](#hallucination-detection)
- [Fact Verification](#fact-verification)
- [Evaluation](#evaluation)

## Installation

### Environment Setup

Create and activate a conda environment:

```bash
conda create -n hd_vs_fv python=3.10
conda activate hd_vs_fv
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
python -m spacy download en_core_web_sm
```

## Dataset Setup

### 2WikiMultihopQA

1. Download the dataset from the [official repository](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1)
2. Extract and move the folder to `data/2wikimultihopqa`

### HotpotQA

```bash
mkdir -p data/hotpotqa
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

### PopQA

```bash
mkdir -p data/popqa
wget -P data/popqa https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv
```

### TriviaQA

```bash
mkdir -p data/triviaqa
wget -P data/triviaqa https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
tar -xvzf data/triviaqa/triviaqa-unfiltered.tar.gz -C data/triviaqa
```

### Natural Questions (NQ)

Download the `NQ-open.efficientqa.dev.1.1.jsonl` file from the [Google Research repository](https://github.com/google-research-datasets/natural-questions) and place it in `data/nq/`.

## Hallucination Detection

### Quick Start

Run hallucination detection with methods that don't require additional training:

```bash
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --judge_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "popqa" \
    --dataset_type "total"
```

**Available training-free methods:** `mqag`, `bertscore`, `ngram`, `nli`, `ptrue`, `lnpp`, `lnpe`, `semantic_entropy`, `seu`, `sindex`

### Methods Requiring Training

#### EUBHD

```bash
python -m scripts.eubhd.count --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

#### MIND

1. Download the wiki dataset from the [official repository](https://github.com/oneal2000/MIND/tree/main)
2. Process using scripts in `scripts/mind/` in the following order:
   - `generate`
   - `extract`
   - `train`

#### SAPLMA

1. Download the dataset from [azariaa.com](azariaa.com/Content/Datasets/true-false-dataset.zip)
2. Process using scripts in `scripts/saplma/` in the following order:
   - `extract`
   - `train`

## Fact Verification

Fact verification requires hallucination detection results. For standalone fact verification, use `--detect_methods lnpp` during hallucination detection to minimize processing time.

### Prerequisites

#### Setup BM25 for Retrieval

1. **Download Wikipedia dump:**

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
cd data/dpr
gzip -d psgs_w100.tsv.gz
cd ../..
```

2. **Setup Elasticsearch indexing:**

```bash
cd data
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0
nohup bin/elasticsearch &  # Run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

#### Train BERT Classifier

1. **Generate training data:**

```bash
python -m scripts.bert.generate_data \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --judge_model_name_or_path Qwen/Qwen2.5-32B-Instruct
```

2. **Train the classifier:**

```bash
python -m scripts.bert.train_bert \
    --data_path "data/train/bert/training_data.json" \
    --output_dir "bert_classifier" \
    --retrieval_type "question_only"
```

### Running Fact Verification

```bash
python -m scripts.fact_verification \
    --target_llm_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "popqa" \
    --dataset_type "total" \
    --fv_llm_model_name Qwen/Qwen2.5-32B-Instruct \
    --bert_fv_q_model_dir bert_classifier/fv_model_question_only \
    --bert_fv_qa_model_dir bert_classifier/fv_model_question_answer
```

## Evaluation

Evaluate detection and fact verification results using AUROC scores:

```bash
python src/evaluate.py <path_to_results_jsonl>
```