# Diffing Game: Model-Diffing Methodologies Research Framework

A research framework for exploring model-diffing methodologies and interpretability techniques. This project enables systematic analysis of the effects of finetuning on language models through various diffing methods.

## Overview

This framework consists of two main pipelines:
1. **Finetuning Pipeline**: Train language models on various tasks
2. **Diffing Pipeline**: Analyze differences between base and finetuned models using interpretability techniques

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffing-game
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

Run the complete pipeline with default settings:
```bash
python main.py
```

### Configuration Examples

Run finetuning only:
```bash
python main.py experiment=finetune_only
```

Run diffing analysis only:
```bash
python main.py experiment=diffing_only
```

Override specific configurations:
```bash
python main.py finetune.task=qa finetune.model=qwen3_0.6B diffing.method=example
```

### Multi-run Experiments

Run experiments across multiple configurations:
```bash
python main.py --multirun finetune.task=caps,frenchcaps finetune.model=qwen3_0.6B
```
