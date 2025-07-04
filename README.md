# 🔥 PHLAME Baselines
---

## 🎯 About PHLAME
**PHLAME** (**PH**enotype prediction **L**ive **A**ssessment from **ME**tagenomic sequencing data) is a **standardized 
and comprehensive benchmark** for evaluating methods that predict **host phenotypes from gut metagenomic data**.

PHLAME provides an **open and fair platform** for comparing predictive methods, with the goal of advancing the use of 
metagenomic data in Health research Disease monitoring.

It includes a diverse set of **binary classification tasks**, each consisting of:
- A labeled **training set**
- A **test set** with hidden labels

---

## 📦 This Repository

This repository contains the code used to run the **baseline methods** described in the PHLAME manuscript.

- Baselines are implemented in **Python**
- Scripts and virtual environments are provided to facilitate reproducibility

---
# 🚀 Getting Started

To run the baseline methods provided in this repository, follow these steps:

## 1. Set Up Conda Environments

We provide three separate Conda environment files for different tools. Create the environments by running:

```bash
conda env create -f phlame_general_python_env.yml
conda env create -f phlame_fcnn_python_env.yml
conda env create -f phlame_deepmicro_python_env.yml
```
## 2. Run the Baseline Tools
Use the following script to run all baseline tools on the benchmark tasks:

```bash
bash run_tools_on_tasks.sh <benchmark_datasets_dir>
```
Replace <benchmark_datasets_dir> with the path to the directory containing all benchmark training and test datasets.

## 3. GPU Usage & Compute Settings
Some tools — specifically the Fully Connected Neural Network (FCNN) and DeepMicro — require GPU support for optimal performance.

* Ensure your system has a CUDA-compatible GPU and the required drivers.

* You may need to adapt the code or scripts to your compute environment (e.g., SLURM-based clusters or other schedulers).

# 📚 Want to learn more on PHLAME?

- 🔗 **Official website**: [https://phlame.yassourlab.com/](https://phlame.yassourlab.com/)
- 📄 **Manuscript on bioRxiv**: [https://www.biorxiv.org/content/10.1101/2025.06.12.658885v1](https://www.biorxiv.org/content/10.1101/2025.06.12.658885v1)

---

## Acknowledgments

The DeepMicro code used in this repository was adapted from the original implementation available on [DeepMicro's GitHub Repository](https://github.com/minoh0201/DeepMicro). We would like to thank them for making their code publicly available and therefore making it possible for us to modify and extended it to suit PHLAME's training & test scheme.
