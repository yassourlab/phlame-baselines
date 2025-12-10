# LAMPP Baselines
---

## 🎯 About LAMPP
**LAMPP** (**L**ive **A**ssessment of **M**etagenomics-based tools for host **P**henotype **P**rediction) is a **standardized 
and comprehensive benchmark** for evaluating methods that predict **host phenotypes from gut metagenomic data**.

PHLAME provides an **open and fair platform** for comparing predictive methods, with the goal of advancing the use of 
metagenomic data in Health research Disease monitoring.

It includes a diverse set of **binary classification tasks**, each consisting of:
- A labeled **training set**
- A **test set** with hidden labels

---

## 🚀 Getting Started

This repository contains the code used to run the **baseline methods** described in the LAMPP manuscript. To run the baseline methods provided in this repository, follow these steps:

### 1. Set Up Conda Environments

We provide three separate Conda environment files for different tools. Create the environments by running:

```bash
conda env create -f lampp_general_python_env.yml
conda env create -f lampp_fcnn_python_env.yml
conda env create -f lampp_deepmicro_python_env.yml
```
### 2. Run the Baseline Tools
Use the following script to run all baseline tools on the benchmark tasks:

```bash
bash run_tools_on_tasks.sh <benchmark_datasets_dir>
```
Replace `<benchmark_datasets_dir>` with the path to the directory containing all benchmark training and test datasets.
Paths should be `<benchmark_datasets_dir>/<task_name>_train.csv` and `<benchmark_datasets_dir>/<task_name>_test.csv` for train and test respectively, assuming the task names are `"scz"`, `"ghs"`, `"crc"`, `"ibd"`, `"dmw"`, `"dmnw"`.

### 3. GPU Usage & Compute Settings
The Fully Connected Neural Network (FCNN) and DeepMicro — require GPU support for optimal performance.

* Ensure your system has a CUDA-compatible GPU and the required drivers. 
* You might need to adjust the cuda version loaded in run_tools_on_tasks.sh

## 📚 Want to learn more on LAMPP?

- 🔗 **Official website**: [https://lampp.yassourlab.com/](https://lampp.yassourlab.com/)
- 📄 **Manuscript on bioRxiv**: [https://www.biorxiv.org/content/10.1101/2025.06.12.658885v1](https://www.biorxiv.org/content/10.1101/2025.06.12.658885v1)

---

## Acknowledgments

The DeepMicro code used in this repository was adapted from the original implementation available on [DeepMicro's GitHub Repository](https://github.com/minoh0201/DeepMicro). We would like to thank them for making their code publicly available and therefore making it possible for us to modify and extended it to suit LAMPP's training & test scheme.
