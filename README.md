# Diffing Toolkit: Model Comparison and Analysis Framework
[Work-In-Progress]
A research framework for analyzing differences between language models using interpretability techniques. This project enables systematic comparison of base models and their variants (model organisms) through various diffing methodologies.

Note: The toolkit is based on a heavily modified version of the  [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning) repository, available at [science-of-finetuning/dictionary_learning](https://github.com/science-of-finetuning/dictionary_learning). Although we may eventually merge these repositories, this is currently not a priority due to significant divergence.

## Overview

This framework consists of two main pipelines:
1. **Preprocessing Pipeline**: Extract and cache activations from pre-existing models
2. **Diffing Pipeline**: Analyze differences between models using interpretability techniques

The framework is designed to work with pre-existing model pairs (e.g., base models vs. model organisms) rather than training new models.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/science-of-finetuning/diffing-game
cd diffing-game
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run the complete pipeline (preprocessing + diffing) with default settings:
```bash
python main.py
```

### Pipeline Modes

Run preprocessing only (extract activations):
```bash
python main.py pipeline.mode=preprocessing
```

Run diffing analysis only (assumes activations already exist):
```bash
python main.py pipeline.mode=diffing
```

### Configuration Examples

Analyze specific organism and model combinations:
```bash
python main.py organism=caps model=gemma3_1B
```

Use different diffing methods:
```bash
python main.py diffing/method=kl
python main.py diffing/method=normdiff
```

### Multi-run Experiments

Run experiments across multiple configurations:
```bash
python main.py --multirun organism=caps,roman_concrete model=gemma3_1B
```

Run with different diffing methods:
```bash
python main.py --multirun diffing/method=kl,normdiff
```
