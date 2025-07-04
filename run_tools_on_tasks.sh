#!/bin/bash -e

benchmark_datasets_dir=${1}
timing_output_file='phlame_benchmark_runtimes.csv'

# create output file and write header if it doesn't exist
[ ! -f "$output_file" ] && touch "$output_file" && echo "command,time" > "$output_file"

run_and_time() {
    local cmd="$1"
    local start_time=$(date +%s)  # Seconds
    eval "$cmd"  # Execute command
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))  # Calculate time in seconds
    echo "$cmd,$elapsed" >> "$timing_output_file"
}

# use a for loop to iterate on the tasks task_names_list=("scp" "ghs" "crc" "ibd" "dmw" "dmnw")
task_names_list=("scp" "ghs" "crc" "ibd" "dmw" "dmnw")
for task_name in "${task_names_list[@]}"
do
    echo "Task name: $task_name"

    # # run random forest, xgboost, svm and logistic regression
    echo "Running RF"
    conda activate phlame_general_python_env
    run_and_time "python vanilla_random_forest_and_xgboost.py $task_name $benchmark_datasets_dir random_forest"

    echo "Running XGBoost"
    run_and_time "python vanilla_random_forest_and_xgboost.py $task_name $benchmark_datasets_dir xgboost"

    echo "Running logistic regression"
    run_and_time "python vanilla_random_forest_and_xgboost.py $task_name $benchmark_datasets_dir logistic_regression"

    echo "Running SVM"
    run_and_time "python vanilla_random_forest_and_xgboost.py $task_name $benchmark_datasets_dir svm"
    conda deactivate

    echo "Running SIAMCAT"
    run_and_time "Rscript siamcat_train_and_predict_flow.R $task_name $benchmark_datasets_dir"

    echo "Running DeepMicro"
    cd /cs/usr/netta.barak/lab/Projects/gut_mgx_benchmark/methods_to_benchmark/DeepMicro-benchmark/
    conda activate phlame_deepmicro_python_env
    run_and_time "python run_dm_for_benchmark.py $task_name $benchmark_datasets_dir"
    conda deactivate

    cd /cs/usr/netta.barak/lab/Projects/gut_mgx_benchmark/mgx-benchmark-code/benchmarking_flow
    echo "Running a Fully connected neural network"
    conda activate phlame_fcnn_python_env
    run_and_time "python simple_nns.py $task_name $benchmark_datasets_dir fully_connected"
    conda deactivate

    echo "All methods completed successfully on task $task_name"
done