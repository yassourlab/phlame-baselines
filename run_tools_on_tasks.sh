#!/bin/bash -e
#SBATCH -c 8
#SBATCH -t 6-23
#SBATCH --mem=16G
#SBATCH --array=1-6
#SBATCH --mail-type=FAIL
#SBATCH --gres gpu:1

task_names_list=("scz" "ghs" "crc" "ibd" "dmw" "dmnw")

task_name=${task_names_list[$SLURM_ARRAY_TASK_ID-1]}
benchmark_datasets_dir=$1

if [ -z "$benchmark_datasets_dir" ]; then
    echo "Error: benchmark_datasets_dir parameter is required"
    echo "Usage: sbatch run_tools_on_tasks.sh <benchmark_datasets_dir>"
    exit 1
fi

echo "Task name: $task_name"
echo "Benchmark datasets directory: $benchmark_datasets_dir"

echo "Running sklearn methods"
mamba activate benchmark_jupyter

echo "Running RF"
python -u sklearn_methods.py $task_name $benchmark_datasets_dir random_forest

echo "Running XGBoost"
python -u sklearn_methods.py $task_name $benchmark_datasets_dir xgboost

echo "Running logistic regression"
python -u sklearn_methods.py $task_name $benchmark_datasets_dir logistic_regression

echo "Running SVM"
python -u sklearn_methods.py $task_name $benchmark_datasets_dir svm
mamba deactivate

echo "Running SIAMCAT"
module load R4/4.4.1
Rscript siamcat_train_and_predict_flow.R $task_name $benchmark_datasets_dir

echo "Running DeepMicro"
mamba activate jupyter_deep_micro_cuda10
cd DeepMicro-benchmark
python -u run_dm_for_benchmark.py $task_name $benchmark_datasets_dir
cd ..
mamba deactivate

echo "Running a Fully connected neural network"
module load cuda/12.4.1
mamba activate simple_pytorch
python -u simple_nns.py $task_name $benchmark_datasets_dir fully_connected
mamba deactivate

echo "All methods completed successfully"
