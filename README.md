# LAMPP Baselines
---

## 🎯 About LAMPP
**LAMPP** (**L**ive **A**ssessment of **M**etagenomics-based tools for host **P**henotype **P**rediction) is a **standardized
and comprehensive benchmark** for evaluating methods that predict **host phenotypes from gut metagenomic data**.

LAMPP provides an **open and fair platform** for comparing predictive methods, with the goal of advancing the use of
metagenomic data in health research and disease monitoring.

It includes a diverse set of **binary classification tasks**, each consisting of:
- A labeled **training set**
- A **test set** with hidden labels

---

## 🚀 Getting Started

This repository contains the code used to run the **baseline methods** described in the LAMPP manuscript, along with
the SLURM-based scripts we used to run them on our cluster.

### 1. Download the Benchmark Datasets

Download the benchmark datasets from the [LAMPP website](https://lampp.yassourlab.com/). Each task provides
`<task>_train.csv`, `<task>_test.csv`, and `<task>_test_gt.csv` files (the latter holds the hidden test labels, used
for evaluation). Available tasks:
- `scz` - Schizophrenia prediction
- `ghs` - Gestational health status
- `crc` - Colorectal cancer
- `ibd` - Inflammatory bowel disease
- `dmw` - Delivery mode (western populations)
- `dmnw` - Delivery mode (non-western populations)

### 2. Set Up Conda Environments

We provide four Conda environment files for the different tools. Create the environments by running:

```bash
conda env create -f lampp_general_python_env.yml   # benchmark_jupyter: sklearn methods, DEBIAS-M, MLflow logging
conda env create -f lampp_fcnn_python_env.yml       # simple_pytorch: fully connected NN
conda env create -f lampp_deepmicro_python_env.yml  # jupyter_deep_micro_cuda10: DeepMicro (TensorFlow 1.x, CUDA 10)
conda env create -f lampp_tabpfn_python_env.yml     # tabpfn: TabPFN
```

You'll also need R (4.4.1+) with [SIAMCAT](https://bioconductor.org/packages/release/bioc/html/SIAMCAT.html),
`tidyverse`, `dplyr`, `rlang`, and `jsonlite` installed for the SIAMCAT baseline.

### 3. TabPFN Authentication (one-time setup)

TabPFN requires authentication with Hugging Face to access the gated model `Prior-Labs/tabpfn_2_5`:

1. Create a Hugging Face account at https://huggingface.co/
2. Create a read token at https://huggingface.co/settings/tokens
3. Accept the gated model terms at https://huggingface.co/Prior-Labs/tabpfn_2_5
4. Set the token in your environment, e.g. add to `~/.bashrc`:
   ```bash
   export HF_TOKEN="hf_YourTokenHere"
   ```

`run_gpu_tools.sh` redirects the HuggingFace cache and temp directories to `.cache/tabpfn/` inside the repo to avoid
filling your home directory quota. See https://docs.priorlabs.ai/how-to-access-gated-models for more details.

### 4. Run the Baseline Tools

All launcher scripts were written for a SLURM cluster but can also be run directly with `bash` for local testing
(GPU tools still require a GPU). Each script loads modules and conda environments via Lmod/`mamba` calls that are
specific to our cluster (HUJI) — adjust these for your own environment if needed.

Before running, edit the `BENCHMARK_DATASETS_DIR` variable near the top of `run_gpu_tools.sh`, `run_long_cpu_tools.sh`,
and `run_medium_cpu_tools.sh` to point at the directory from step 1.

**Run everything** (submits one SLURM array job per tool, one array element per task):
```bash
bash run_all_benchmarks.sh
```

**Run a subset of tools and/or tasks:**
```bash
bash run_all_benchmarks.sh --tools "random_forest,xgboost" --benchmark-tasks "crc,ibd"
```

**Run a specific group of tools directly:**
```bash
sbatch run_medium_cpu_tools.sh   # random_forest, xgboost, logistic_regression, svm
sbatch run_long_cpu_tools.sh     # siamcat, debias_m
sbatch run_gpu_tools.sh          # deep_micro, fully_connected, tabpfn
```

#### Available tools

- CPU: `random_forest`, `xgboost`, `logistic_regression`, `svm`, `siamcat`, `debias_m`
- GPU: `deep_micro`, `fully_connected`, `tabpfn`

#### Command-line options

All scripts share argument parsing from `benchmark_args_parser.sh`:

```
--benchmark-tasks TASKS    Comma-separated list of tasks to run (default: all tasks)
--tool TOOL                Single tool to run (used internally for array jobs)
--tools TOOLS              Comma-separated list of tools to run (default: all tools for this script)
--exclude-tasks TASKS      Comma-separated list of tasks to exclude
--exclude-tools TOOLS      Comma-separated list of tools to exclude
--aggregate-only           Run aggregation + MLflow logging only (SIAMCAT/DEBIAS-M, skip per-config jobs)
-h, --help                 Show help message
```

### 5. Output

Results are saved under `benchmarking_outputs/<task>_<model>/<run_type>/`, where `<run_type>` is `default` or
`optimized`:
- `predictions.csv` — `sample_id`, `prediction` (predicted probability) for each test sample
- `train_time.txt` — training time in seconds
- `params_search.csv` — cross-validation results for each hyperparameter combination tried

### 6. MLflow Logging

After each tool finishes, `log_results_to_mlflow.py` logs predictions, metrics, and ROC/PR curves to MLflow. By
default, results are logged to a local `./mlflow` directory; set the `MLFLOW_TRACKING_URI` environment variable to
point at a shared tracking server or directory instead:

```bash
export MLFLOW_TRACKING_URI="/path/to/shared/mlflow"
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

### 7. Downloading Raw Sequencing Data

`download_samples.sh` downloads the raw FASTQ files for samples listed in a task's CSV file via SLURM array jobs
(using `fasterq-dump`/`prefetch`/`esearch` with fallbacks):

```bash
sbatch --array=1-N download_samples.sh /path/to/task_data.csv /output/dir
```
where `N` is the number of rows in the CSV file.

### 8. Reproducibility

All models use a fixed random seed (`SEED = 17`, see `benchmark_utils.py`) for cross-validation splits and model
initialization.

## Monitoring

```bash
squeue -u $USER
tail -f logs/<tool>/<task>/*.out
```

## Troubleshooting

**TabPFN authentication error**
```
RuntimeError: Failed to download TabPFN ModelVersion.V2_5 model
HuggingFace authentication error downloading from 'Prior-Labs/tabpfn_2_5'
```
Set up your Hugging Face token (see step 3 above).

**GPU not available**
Run GPU tools via `run_gpu_tools.sh` (or `run_all_benchmarks.sh`), which request a GPU from SLURM via `#SBATCH --gres=gpu:1`.

**Environment not found / module not found**
Make sure you've created all four conda environments (step 2) with the names expected by the scripts:
`benchmark_jupyter`, `simple_pytorch`, `jupyter_deep_micro_cuda10`, `tabpfn`.

## 📚 Want to learn more on LAMPP?

- 🔗 **Official website**: [https://lampp.yassourlab.com/](https://lampp.yassourlab.com/)
- 📄 **Manuscript on bioRxiv**: [https://www.biorxiv.org/content/10.1101/2025.06.12.658885](https://www.biorxiv.org/content/10.1101/2025.06.12.658885)

---

## Acknowledgments

The DeepMicro code used in this repository was adapted from the original implementation available on [DeepMicro's GitHub Repository](https://github.com/minoh0201/DeepMicro). We would like to thank them for making their code publicly available and therefore making it possible for us to modify and extend it to suit LAMPP's training & test scheme.
