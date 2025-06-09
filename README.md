# Control Vector Experiment Pipeline

A comprehensive experimental framework for analyzing and manipulating control vectors in transformer language models. Implements five core experiments that explore different mechanisms for neural control in GPT-2 and TinyLlama models.

## Overview

This repository demonstrates practical techniques for steering model outputs through hidden state manipulation, attention targeting, and dynamic control strategies.

## Core Experiments

### 🎯 Experiment 1: Layer-Wise Token Trajectory Stabilization
Detects and corrects probability drift across transformer layers. Monitors target token probabilities and applies control vectors when they drop below threshold.

```bash
python experiments.py --experiments 1
```

### 🌪️ Experiment 2: Entropy-Guided Control Vector Calibration
Dynamically adjusts control strength based on model uncertainty using entropy: `alpha = base_alpha * (entropy / target_entropy)`.

```bash
python experiments.py --experiments 2
```

### 🔍 Experiment 3: Attention Attribution-Driven Control
Identifies and targets the most important attention heads for specific predictions, achieving 97% efficiency by controlling only top 3 heads.

```bash
python experiments.py --experiments 3
```

### 🎨 Experiment 4: Head-Cluster Extraction for Semantic Control
Learns control vectors for style transfer by comparing neural patterns between different writing styles.

```bash
python experiments.py --experiments 4
```

### 🧠 Experiment 5: Dynamic Multi-Signal Control Vector Injection
Uses neural controller to orchestrate multiple control signals (probability, entropy, attribution, cluster scores).

```bash
python experiments.py --experiments 5
```

## Quick Start

### Installation
```bash
# Install dependencies
pip install torch transformers numpy scikit-learn

# Run experiments
./run_experiments.sh  # TinyLlama with experiments 1 & 2
```

### Basic Usage
```bash
# Run all experiments with GPT-2
python experiments.py --experiments all

# Run specific experiments with TinyLlama
python experiments.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --experiments 1 2

# Use different model
python experiments.py --model gpt2-medium --experiments 2

# Custom logging
python experiments.py --experiments 4 --log-level DEBUG --log-dir custom_logs
```

## Supported Models

- **GPT-2**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **TinyLlama**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`

## Key Features

### 🔬 **Comprehensive Analysis**
- Real model predictions at every step
- Before/after control comparisons
- Quantified effectiveness metrics
- Layer-by-layer behavioral analysis

### 🎮 **Multiple Control Methods**
- Hidden state manipulation
- Attention head targeting  
- Style transfer vectors
- Logit boosting and embedding steering

### 📊 **Rich Logging**
- Detailed console output with emoji indicators
- Timestamped log files with full experiment traces
- Probability trajectories and entropy profiles
- Generation comparisons

## Understanding the Output

Each experiment follows this flow:
1. **🤖 Model Baseline**: Natural predictions
2. **🔍 Analysis Phase**: Internal state examination
3. **📝 Control Application**: Intervention technique
4. **📊 Results Evaluation**: Effectiveness comparison
5. **🎯 Generation Testing**: Real text generation effects

### Success Indicators
- **✅ Target Improved**: Increased target token probability
- **✅ Prediction Changed**: Shifted model's top prediction
- **✅ Generation Success**: Target tokens in generated text

## Research Applications

### 🎯 **Model Steering**
- Bias mitigation in language generation
- Content filtering and safety controls
- Style adaptation for different audiences

### 🔬 **Model Understanding**
- Mechanistic interpretability of transformer layers
- Attention head function analysis
- Information flow visualization

### 🛠️ **Control Engineering**
- Efficient intervention strategies
- Adaptive control strength calibration
- Multi-objective optimization

## Example Results

```
🤖 MODEL'S ACTUAL PREDICTION:
  Final prediction: ' is' (P=0.8234)
  Target token probability: 0.0123

🚨 DRIFT DETECTED in 3/12 layers: [4, 7, 9]

📝 APPLYING CONTROL:
  - Control layer: 4
  - Original P(target): 0.0089
  - New P(target): 0.2456
  - Improvement: +0.2367 ✅

🎯 TESTING GENERATION:
  - Original: "The quick brown fox is running"
  - Controlled: "The quick brown fox jumps over"
  - SUCCESS: Target token 'jumps' appeared! ✅
```

## Configuration

### Control Parameters
- `threshold`: Probability threshold for drift detection (default: 0.2)
- `alpha`: Control vector strength (default: 0.5)
- `H_target`: Target entropy for normalization (default: 5.0)
- `top_n`: Number of attention heads to target (default: 3)

### Custom Prompts
```python
prompts = ["Your custom prompt here"]
gold_ids = [tokenizer.encode(" target_word")[0]]
```

### Script Options
```bash
./run_experiments.sh --help  # View all options
./run_experiments.sh --model gpt2-medium --experiments 1 3 5 --log-level DEBUG
```
