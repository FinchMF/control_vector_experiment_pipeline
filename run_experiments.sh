#!/bin/bash

# Control Vector Experiment Pipeline Runner
# ========================================
# 
# This script provides convenient commands for running control vector experiments
# with various models and configurations. All available options are documented below.
#
# Authors: Research Team
# Date: 2024

# Script configuration
set -e  # Exit on any error
set -u  # Exit on undefined variables

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display help
show_help() {
    echo "Control Vector Experiment Pipeline Runner"
    echo "========================================"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Available Models:"
    echo "  --model gpt2                           # GPT-2 base model (124M parameters)"
    echo "  --model gpt2-medium                    # GPT-2 medium model (355M parameters)"
    echo "  --model gpt2-large                     # GPT-2 large model (774M parameters)"
    echo "  --model gpt2-xl                        # GPT-2 XL model (1.5B parameters)"
    echo "  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0           # TinyLlama Chat model (1.1B parameters)"
    echo "  --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  # TinyLlama base model"
    echo ""
    echo "Experiment Selection:"
    echo "  --experiments all                      # Run all 5 experiments (default)"
    echo "  --experiments 1                        # Layer-Wise Token Trajectory Stabilization"
    echo "  --experiments 2                        # Entropy-Guided Control Vector Calibration"
    echo "  --experiments 3                        # Attention Attribution-Driven Control"
    echo "  --experiments 4                        # Head-Cluster Extraction for Semantic Control"
    echo "  --experiments 5                        # Dynamic Multi-Signal Control Vector Injection"
    echo "  --experiments 1 2                      # Run experiments 1 and 2"
    echo "  --experiments 1 2 3 4 5                # Run all experiments explicitly"
    echo ""
    echo "Logging Options:"
    echo "  --log-level DEBUG                      # Detailed debug information"
    echo "  --log-level INFO                       # General information (default)"
    echo "  --log-level WARNING                    # Warnings and errors only"
    echo "  --log-level ERROR                      # Errors only"
    echo "  --log-dir logs                         # Directory for log files (default: logs)"
    echo "  --log-dir /path/to/custom/logs         # Custom log directory"
    echo ""
    echo "Quick Commands:"
    echo "  $0                                     # Run all experiments with GPT-2"
    echo "  $0 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --experiments 1 2  # TinyLlama with experiments 1&2"
    echo "  $0 --experiments 4 --log-level DEBUG  # Style control experiment with debug logging"
    echo "  $0 --model gpt2-medium --experiments 5 # Neural controller experiment with medium GPT-2"
    echo ""
    echo "Example Experiment Descriptions:"
    echo "  Experiment 1: Analyzes probability drift across transformer layers"
    echo "  Experiment 2: Uses entropy to dynamically adjust control strength"
    echo "  Experiment 3: Identifies most important attention heads for control"
    echo "  Experiment 4: Extracts style transfer vectors between text types"
    echo "  Experiment 5: Uses neural network to orchestrate multiple control signals"
    echo ""
    echo "Output:"
    echo "  - Console output with experiment progress and results"
    echo "  - Detailed log file saved to specified log directory"
    echo "  - Log files are timestamped: control_vector_experiments_YYYYMMDD_HHMMSS.log"
    echo ""
}

# Default values
MODEL="gpt2"
EXPERIMENTS="all"
LOG_LEVEL="INFO"
LOG_DIR="logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="experiments.py"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --experiments|-e)
            shift
            EXPERIMENTS=""
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                EXPERIMENTS="$EXPERIMENTS $1"
                shift
            done
            EXPERIMENTS=$(echo $EXPERIMENTS | xargs)  # Trim whitespace
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! "$LOG_LEVEL" =~ ^(DEBUG|INFO|WARNING|ERROR)$ ]]; then
    print_error "Invalid log level: $LOG_LEVEL. Must be DEBUG, INFO, WARNING, or ERROR"
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$SCRIPT_DIR/$PYTHON_SCRIPT" ]]; then
    print_error "Python script not found: $SCRIPT_DIR/$PYTHON_SCRIPT"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Display configuration
print_info "Control Vector Experiment Configuration:"
echo "  Model: $MODEL"
echo "  Experiments: $EXPERIMENTS"
echo "  Log Level: $LOG_LEVEL"
echo "  Log Directory: $LOG_DIR"
echo "  Script Location: $SCRIPT_DIR/$PYTHON_SCRIPT"
echo ""

# TinyLlama 1.1B Chat v1.0 with experiments 1 and 2
# ==================================================
# This is the main command for running the first two experiments
# with TinyLlama Chat model as requested

print_info "Starting Control Vector Experiments..."
print_info "Running gpt2 with experiments 1 and 2"

# Check if we have the required Python dependencies
print_info "Checking Python dependencies..."
python3 -c "import torch, transformers, sklearn, numpy" 2>/dev/null || {
    print_error "Missing required Python packages. Please install:"
    echo "  pip install torch transformers scikit-learn numpy"
    exit 1
}

# Execute the main command
cd "$SCRIPT_DIR"

print_info "Executing command:"
echo "python3 experiments.py --model gpt2--experiments 1 2 --log-level $LOG_LEVEL --log-dir $LOG_DIR"
echo ""

# Run the experiment
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
if python3 experiments.py \
    --model "gpt2"\
    --experiments 1 2 \
    --log-level "$LOG_LEVEL" \
    --log-dir "$LOG_DIR"; then
    
    print_success "Experiments completed successfully!"
    
    # Find the most recent log file
    LATEST_LOG=$(ls -t "$LOG_DIR"/control_vector_experiments_*.log 2>/dev/null | head -n1)
    if [[ -n "$LATEST_LOG" ]]; then
        print_info "Detailed log saved to: $LATEST_LOG"
        print_info "Log file size: $(du -h "$LATEST_LOG" | cut -f1)"
    fi
    
    echo ""
    print_info "Experiment Summary:"
    echo "  ✅ Experiment 1: Layer-Wise Token Trajectory Stabilization"
    echo "  ✅ Experiment 2: Entropy-Guided Control Vector Calibration"
    echo ""
    print_info "Next steps:"
    echo "  - Review the log file for detailed results"
    echo "  - Run additional experiments: $0 --experiments 3 4 5"
    echo "  - Try different models: $0 --model gpt2-medium"
    
else
    print_error "Experiments failed! Check the error output above."
    exit 1
fi

# Additional example commands (commented out)
# ===========================================

# Example: Run all experiments with GPT-2 medium
# python3 experiments.py --model gpt2-medium --experiments all --log-level INFO --log-dir logs

# Example: Run style control experiment with debug logging
# python3 experiments.py --model gpt2 --experiments 4 --log-level DEBUG --log-dir logs

# Example: Run attention attribution experiment only
# python3 experiments.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --experiments 3 --log-level INFO --log-dir logs

# Example: Run neural controller experiment with custom log directory
# python3 experiments.py --model gpt2-large --experiments 5 --log-level INFO --log-dir /tmp/experiment_logs

# Example: Run multiple experiments with TinyLlama base model
# python3 experiments.py --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --experiments 1 3 5 --log-level DEBUG --log-dir logs

# Batch processing examples:
# =========================

# Run all experiments across different models
# for model in gpt2 gpt2-medium TinyLlama/TinyLlama-1.1B-Chat-v1.0; do
#     echo "Running experiments with $model"
#     python3 experiments.py --model "$model" --experiments all --log-dir "logs/$model"
# done

# Run specific experiment across models for comparison
# for model in gpt2 TinyLlama/TinyLlama-1.1B-Chat-v1.0; do
#     echo "Running experiment 4 (style control) with $model"
#     python3 experiments.py --model "$model" --experiments 4 --log-dir "logs/style_comparison"
# done

print_info "Script completed. Use --help for more options."
