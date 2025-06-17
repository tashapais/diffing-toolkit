#!/bin/bash

# Dynamic script for running preprocessing experiments
# Usage: ./preprocessing.sh <organism> <method> [additional_args...]
# Example: ./preprocessing.sh kansas_abortion crosscoder pipeline.mode=preprocessing

if [ $# -lt 2 ]; then
    echo "Usage: $0 <organism> <method> [additional_args...]"
    echo "Example: $0 kansas_abortion crosscoder infrastructure=runpod"
    exit 1
fi

ORGANISM=$1
METHOD=$2
shift 2  # Remove first two arguments

# Run the command with dynamic arguments
python main.py diffing/method=$METHOD infrastructure=runpod organism=$ORGANISM pipeline.mode=preprocessing "$@"

