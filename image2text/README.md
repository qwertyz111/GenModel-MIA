
## 🚀 Quick Start

### 1. Environment Setup

Use Anaconda to quickly set up the required Python environment:

```bash
conda env create -f environment.yml
conda activate llava
```

### 2. Data and Model Preparation

This project depends on baseline resources from multimodal MIA research. Please follow these steps:

*   **Reference Repository**: [YukeHu/vlm_mia](https://github.com/YukeHu/vlm_mia)
*   **Dataset & Models**: Download the relevant datasets and model weights as instructed in the reference repository.
*   **Conversation Generation**: Use the `Conversation Generation` module in the reference repository to generate the required dialogue data.

### 3. Code Configuration

Before running the scripts, please update the path configurations in the code:

```python
data = load_data('path/to/your/conversation_file')
model_name = 'path/to/your/encoder'
```

## 📂 Project Structure

The core logic of the project is organized into the following scripts:

### Data Generation & Metrics (`data_*.py`)
Organized by different similarity/distance metrics, including synonym-enhanced and differential variants:
*   **Cosine Similarity**: `data_consine.py`, `data_consine_with_synonyms.py`, etc.
*   **L2 Distance**: `data_l2.py`, `data_l2_with_synonyms.py`, etc.
*   **Wasserstein Distance**: `data_wasserstein.py`, `data_wasserstein_with_synonyms.py`, etc. (Note: The `data_wasserstein_with_synonyms.py` and its differential variant are baseline methods).
*   **Ablation Study**: `data_ablation.py` - Used for validating "optimal z" (geometric median).
*   **Threshold Comparison**: Files ending in `_diff.py` (e.g., `data_consine_diff.py`) are used to compare results under different thresholds.


### Text Classifiers (`text_classifier*.py`)
Used for evaluating and executing MIA attacks:
*   `text_classiffier.py`: Basic implementation of the text classifier.
*   `text_classifier_synonym.py`: Text classifier with synonym-based robustness enhancement.

### Baseline (`shawdow_member*.py`)
These scripts implement membership inference attacks using the shadow model approach (Baseline Methods):
*   `shawdow_member.py`: Basic shadow member attack implementation.
*   `shawdow_member_synonyms.py`: Shadow member attack with synonym-based augmentation.

## 🛠 Tech Stack

*   **Language**: Python 3.10.14
*   **Core Frameworks**: PyTorch 2.1.2, Transformers 4.37.2
*   **Multimodal Support**: LLaVA (Large Language-and-Vision Assistant)
*   **Others**: FastAPI, Gradio (for API and interactive interfaces)

## 📄 License

Please refer to the license of the original project [vlm_mia](https://github.com/YukeHu/vlm_mia).

