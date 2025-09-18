#!/bin/bash

# NS-MOE Training Script for Time Series Forecasting
# This script trains NS-MOE models on all major time series forecasting datasets
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, Exchange

set -e  # Exit on any error

# Configuration
GPU_ID=0
LOG_DIR="./training_logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Create log directory
mkdir -p $LOG_DIR

echo "========================================"
echo "Starting NS-MOE Training Pipeline"
echo "Timestamp: $TIMESTAMP"
echo "GPU ID: $GPU_ID"
echo "========================================"

# Function to run training with logging
train_model() {
    local config_file=$1
    local model_name=$2
    local dataset_name=$3
    local log_file="$LOG_DIR/${model_name}_${dataset_name}_${TIMESTAMP}.log"
    
    echo "Training $model_name on $dataset_name..."
    echo "Config: $config_file"
    echo "Log: $log_file"
    
    python experiments/train.py -c "$config_file" --gpus $GPU_ID 2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully trained $model_name on $dataset_name"
    else
        echo "❌ Failed to train $model_name on $dataset_name"
        echo "Check log file: $log_file"
    fi
    echo "----------------------------------------"
}

# Function to run evaluation with logging
evaluate_model() {
    local config_file=$1
    local checkpoint_path=$2
    local model_name=$3
    local dataset_name=$4
    local log_file="$LOG_DIR/eval_${model_name}_${dataset_name}_${TIMESTAMP}.log"
    
    echo "Evaluating $model_name on $dataset_name..."
    echo "Config: $config_file"
    echo "Checkpoint: $checkpoint_path"
    echo "Log: $log_file"
    
    if [ -f "$checkpoint_path" ]; then
        python experiments/evaluate.py --config "$config_file" --checkpoint "$checkpoint_path" --gpus $GPU_ID 2>&1 | tee "$log_file"
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully evaluated $model_name on $dataset_name"
        else
            echo "❌ Failed to evaluate $model_name on $dataset_name"
        fi
    else
        echo "⚠️  Checkpoint not found: $checkpoint_path"
        echo "Skipping evaluation for $model_name on $dataset_name"
    fi
    echo "----------------------------------------"
}

echo "Starting NS-MOE Model Training"
echo "==============================="

# NS-MOE MODELS TRAINING
echo "Training NS-MOE models..."
train_model "baselines/NSMOE/NSMOE_ETTh1.py" "NSMOE" "ETTh1"
train_model "baselines/NSMOE/NSMOE_ETTh2.py" "NSMOE" "ETTh2"
train_model "baselines/NSMOE/NSMOE_ETTm1.py" "NSMOE" "ETTm1"
train_model "baselines/NSMOE/NSMOE_ETTm2.py" "NSMOE" "ETTm2"
train_model "baselines/NSMOE/NSMOE_Electricity.py" "NSMOE" "Electricity"
train_model "baselines/NSMOE/NSMOE_Traffic.py" "NSMOE" "Traffic"
train_model "baselines/NSMOE/NSMOE_Weather.py" "NSMOE" "Weather"
train_model "baselines/NSMOE/NSMOE_Exchange.py" "NSMOE" "Exchange"

# Note: MSNSMOE removed - using only NSMOE

echo ""
echo "Starting NS-MOE Model Evaluation"
echo "================================="

# NS-MOE EVALUATION PHASE
echo "Evaluating trained NS-MOE models..."

# Define checkpoint paths (adjust these based on your actual checkpoint structure)
CHECKPOINT_BASE="./checkpoints"

# NS-MOE Evaluations (all datasets)
evaluate_model "baselines/NSMOE/NSMOE_ETTh1.py" \
               "$CHECKPOINT_BASE/NSMOE/ETTh1_*/best_val_MAE.pth" \
               "NSMOE" "ETTh1"

evaluate_model "baselines/NSMOE/NSMOE_ETTh2.py" \
               "$CHECKPOINT_BASE/NSMOE/ETTh2_*/best_val_MAE.pth" \
               "NSMOE" "ETTh2"

evaluate_model "baselines/NSMOE/NSMOE_ETTm1.py" \
               "$CHECKPOINT_BASE/NSMOE/ETTm1_*/best_val_MAE.pth" \
               "NSMOE" "ETTm1"

evaluate_model "baselines/NSMOE/NSMOE_ETTm2.py" \
               "$CHECKPOINT_BASE/NSMOE/ETTm2_*/best_val_MSE.pth" \
               "NSMOE" "ETTm2"

evaluate_model "baselines/NSMOE/NSMOE_Electricity.py" \
               "$CHECKPOINT_BASE/NSMOE/Electricity_*/best_val_MAE.pth" \
               "NSMOE" "Electricity"

evaluate_model "baselines/NSMOE/NSMOE_Traffic.py" \
               "$CHECKPOINT_BASE/NSMOE/Traffic_*/best_val_MSE.pth" \
               "NSMOE" "Traffic"

evaluate_model "baselines/NSMOE/NSMOE_Weather.py" \
               "$CHECKPOINT_BASE/NSMOE/Weather_*/best_val_MSE.pth" \
               "NSMOE" "Weather"

evaluate_model "baselines/NSMOE/NSMOE_Exchange.py" \
               "$CHECKPOINT_BASE/NSMOE/Exchange_*/best_val_MAE.pth" \
               "NSMOE" "Exchange"

echo ""
echo "======================================="
echo "NS-MOE Training Pipeline Complete!"
echo "======================================="
echo "Timestamp: $TIMESTAMP"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Summary:"
echo "- NS-MOE models trained: NSMOE (all 8 datasets)"
echo "- Datasets covered: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, Exchange"
echo "- Check individual log files for detailed results"
echo ""

# Optional: Generate a summary report
echo "Generating training summary..."
echo "Training Summary - $TIMESTAMP" > "$LOG_DIR/training_summary_${TIMESTAMP}.txt"
echo "=============================================" >> "$LOG_DIR/training_summary_${TIMESTAMP}.txt"
echo "" >> "$LOG_DIR/training_summary_${TIMESTAMP}.txt"

# Count successful/failed trainings
echo "Training Results:" >> "$LOG_DIR/training_summary_${TIMESTAMP}.txt"
grep -l "✅ Successfully trained" $LOG_DIR/*_${TIMESTAMP}.log | wc -l | xargs echo "Successful trainings:" >> "$LOG_DIR/training_summary_${TIMESTAMP}.txt"
grep -l "❌ Failed to train" $LOG_DIR/*_${TIMESTAMP}.log | wc -l | xargs echo "Failed trainings:" >> "$LOG_DIR/training_summary_${TIMESTAMP}.txt"

echo "Training pipeline completed! Check the summary at: $LOG_DIR/training_summary_${TIMESTAMP}.txt"
