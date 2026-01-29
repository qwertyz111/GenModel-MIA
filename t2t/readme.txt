SPV-MIA: Sample Profiling–Based Membership Inference Attacks on Generative Models

This repository contains the implementation of SPV-MIA, a sample-profiling-based membership inference attack framework for large generative language models.

The code supports both training-based and computation-based membership inference attacks across multiple datasets and generative models.

Overview

SPV-MIA is a unified framework that:

Constructs sample-level attack signals from generative model behaviors

Supports training-based, computation-based, and cross-model MIA

Evaluates robustness under partial unlearning, zero unlearning, and DP-SGD defenses

Repository Structure

.
├── attack/ Core attack implementations
├── baseline/ Baseline MIA methods
├── dif_mia/ Differential attack variants
├── DPSGD/ DP-SGD defense experiments
├── Ablation/ Ablation studies
├── configs/ Dataset and model configs
├── data/ Processed datasets (not included)
├── ft_llms/ Fine-tuned LLMs (local path)
├── results/ Raw experimental results
├── results_figs/ Generated figures

├── attack.py Main attack entry
├── mia_unified_36runs.py Unified experiment runner
├── mia_training_based_*.py Training-based MIA scripts
├── mia_computation_based.py Computation-based MIA
├── mia.py Improved computation-based attack

├── requirements.txt
└── README.txt

Environment Setup

Create environment

conda create -n spv_mia python=3.9 -y
conda activate spv_mia

Install dependencies

pip install -r requirements.txt

Note: PyTorch and Transformers versions are sensitive.
CUDA 11.7+ and PyTorch >= 1.13 are recommended.

Running Experiments
Training-Based MIA

Example (AGNews):

python mia_training_based_agnews.py

Supported datasets:

AGNews

WikiText-103

XSum


Unified Evaluation (Recommended)

Runs 36 attack configurations automatically:

python mia_unified_36runs.py


Metrics include:

AUC

ASR

TPR at 1% FPR

Defense Settings

Supported defenses:

Partial Unlearning

Zero Unlearning

DP-SGD (see DPSGD/ directory)

DP-SGD experiments allow evaluation under different privacy budgets.

Datasets and Models

Due to license constraints, datasets and fine-tuned models are not included.

Expected directory structure:

data/
ft_llms/

Refer to configs/ for dataset preprocessing and experiment settings.