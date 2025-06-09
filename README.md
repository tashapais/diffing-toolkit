# Diffing Game: Model Comparison and Analysis Framework

A research framework for analyzing differences between language models using interpretability techniques. This project enables systematic comparison of base models and their variants (model organisms) through various diffing methodologies.

## Overview

This framework consists of two main pipelines:
1. **Preprocessing Pipeline**: Extract and cache activations from pre-existing models
2. **Diffing Pipeline**: Analyze differences between models using interpretability techniques

The framework is designed to work with pre-existing model pairs (e.g., base models vs. model organisms) rather than training new models.

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

Override preprocessing settings:
```bash
python main.py preprocessing.max_samples_per_dataset=50000 preprocessing.layers=[0.5,0.75]
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
