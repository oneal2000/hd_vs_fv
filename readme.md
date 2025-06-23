# Hallucination Detection vs Fact Verification

This repository provides implementations for hallucination detection and fact verification methods for large language models (LLMs). The project includes multiple detection approaches and evaluation frameworks across various QA datasets.

## Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Hallucination Detection](#hallucination-detection)
- [Fact Verification](#fact-verification)
- [Evaluation](#evaluation)

## Installation

```bash
# Create environment
conda create -n hd_vs_fv python=3.10
conda activate hd_vs_fv

# Install dependencies
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
    --dataset_type "total" \
    --detect_methods lnpp
```

**Available training-free methods:** 
- `lnpp`
- `lnpe`
- [`SelfCheckGPT`](http://arxiv.org/abs/2303.08896)
  - `mqag`
  - `bertscore`
  - `ngram`
  - `nli`
- [`ptrue`](http://arxiv.org/abs/2207.05221)
- [`semantic_entropy`](https://www.nature.com/articles/s41586-024-07421-0)
- [`seu`](http://arxiv.org/abs/2410.22685)
- [`sindex`](http://arxiv.org/abs/2503.05980)


### Methods Requiring Training

#### [`EUBHD`](http://arxiv.org/abs/2311.13230)

```bash
# Generate token frequency statistics
python -m scripts.eubhd.count --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

After training the EUBHD method, you can evaluate it using the following command:

```bash
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --detect_methods eubhd \
    --dataset_name "2wikimultihopqa" \
    --dataset_type "bridge_comparison" \
    --eubhd_idf_path eubhd_idf/token_idf_Llama-3.1-8B-Instruct.pkl
```

#### [`SAPLMA`](https://arxiv.org/abs/2304.13734v2)

```bash
# 1. Extract features from last layer
python -m scripts.saplma.extract_features \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --input_dir_path training_data/SAPLMA \
    --output_dir_path saplma/Llama-3.1-8B-Instruct_-1/data \

# 2. Train probe
python -m scripts.saplma.train_probe \
    --embedding_dir_path saplma/Llama-3.1-8B-Instruct_-1/data \
    --output_probe_path saplma/Llama-3.1-8B-Instruct_-1/probe.pt
```


After training the SAPLMA method, you can evaluate it using the following command:

```bash
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --detect_methods saplma \
    --dataset_name "2wikimultihopqa" \
    --dataset_type "bridge_comparison" \
    --saplma_probe_path saplma/Llama-3.1-8B-Instruct_-1/probe.pt
```

#### [`MIND`](http://arxiv.org/abs/2403.06448)

```bash
# 1. Generate training data
python -m scripts.mind.generate_data \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --wiki_data_dir training_data/MIND \
    --output_dir mind/Llama-3.1-8B-Instruct/text_data

# 2. Extract internal features
python -m scripts.mind.extract_features \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --generated_data_dir mind/Llama-3.1-8B-Instruct/text_data \
    --output_feature_dir mind/Llama-3.1-8B-Instruct/feature_data

# 3. Train classifier
python -m scripts.mind.train_mind \
    --feature_dir mind/Llama-3.1-8B-Instruct/feature_data  \
    --output_classifier_dir mind/Llama-3.1-8B-Instruct/classifier \
```

After training the MIND method, you can evaluate it using the following command:

```bash
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --detect_methods mind \
    --dataset_name "2wikimultihopqa" \
    --dataset_type "bridge_comparison" \
    --mind_classifier_path mind/Llama-3.1-8B-Instruct/classifier/mind_classifier_best.pt
```

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
    --judge_model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --data_path bert/training_data.json
```

2. **Train the classifier:**

```bash
python -m scripts.bert.train_bert \
    --output_dir "bert_classifier" \
    --retrieval_type "question_only"

python -m scripts.bert.train_bert \
    --output_dir "bert_classifier" \
    --retrieval_type "question_answer"
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
python -m scripts.evaluation \
    --model_name_or_path  meta-llama/Llama-3.1-8B-Instruct \
    --judge_model_name_or_path  Qwen/Qwen2.5-32B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total"
```

You will get the results in `results/<model_name>/<dataset_name>/<dataset_type>/evaluation_summary.json`

Example:

```json
{
  "fv_BERT_Q": {
    "method": "fv_BERT_Q",
    "total_items": 500,
    "valid_scores": 500,
    "auroc": 0.7215166666666666,
    "positive_cases": 200,
    "negative_cases": 300
  },
    "lnpe": {
    "method": "lnpe",
    "total_items": 500,
    "valid_scores": 500,
    "auroc": 0.7524249999999998,
    "positive_cases": 200,
    "negative_cases": 300
  }
}
```