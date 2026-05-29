# Legend at ArAIEval Shared Task: Persuasion Technique Detection using a Language-Agnostic Text Representation Model

This repository contains the official system description implementation and code framework developed by team **Legend** for **Task 1 (Subtask A) of the ArAIEval Shared Task**, presented at the **ArabicNLP 2023** conference (co-located with EMNLP 2023).

The project introduces an end-to-end transformer fine-tuning framework using **XLM-RoBERTa** to detect structural persuasion and propaganda techniques embedded within multi-genre Arabic texts, including social media tweets and mainstream news articles.

---

##  Abstract

In this paper, we share our best performing submission to the Arabic AI Tasks Evaluation Challenge (ArAIEval) at ArabicNLP 2023. Our focus was on Task 1, which involves identifying persuasion techniques in excerpts from tweets and news articles. The persuasion technique in Arabic texts was detected using a training loop with XLM-RoBERTa, a language-agnostic text representation model. This approach proved to be potent, leveraging fine-tuning of a multilingual language model. In our evaluation of the test set, we achieved a micro F1 score of 0.64 for subtask A of the competition.

---

##  Task & Data Framing

The challenge focuses on identifying intent manipulation across highly erratic social threads and structured journalistic syntax:

* **Subtask A:** A binary classification task. Given a text snippet (tweet or news paragraph), the model must predict whether it contains *at least one* fine-grained persuasion/propaganda technique (`true`) or remains completely impartial and purely informative (`false`).
* **Linguistic Challenge:** Arabic features profound dialectal variations (especially on Twitter) and a complex morphological structure. Traditional models often require heavy lemmatization or language-specific roots. This pipeline uses a language-agnostic embedding approach to capture semantic footprints across scripts without rigid preprocessing.

---

##  Model Architecture

The core pipeline bypasses traditional feature engineering by adapting massive cross-lingual transfer learning:

* **Base Architecture:** Fine-tuned `xlm-roberta-base` / `xlm-roberta-large` to map Arabic text strings into a shared multilingual vector space.
* **Training Loop:** Leverages a customized sequence classification head optimized with cross-entropy loss over unbalanced binary target distributions.
* **Tokenization Protection:** Employs byte-level Byte-Pair Encoding (BPE) to naturally handle out-of-vocabulary slang words, mixed-script characters, and typos found in Arabic social media feeds.

---

##  Shared Task Results

Evaluated on the official blind test set arrays provided by the ArAIEval organizers, our best-performing transformer configuration achieved a competitive **Micro F1-Score of 0.64** for Subtask A.

---

##  Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* Scikit-Learn
* Pandas, NumPy

### Installation

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt

```

### Usage

1. **Preprocess and Clean Multi-Genre Text:**
```bash
python src/data_prep.py --input_dir ./data/araieval_task1

```



```
2. **Execute Fine-Tuning Optimization Loop:**
   ```bash
   python src/train_persuasion_detector.py --model xlm-roberta-base --epochs 4 --batch_size 32 --lr 2e-5

```

3. **Generate Official Submission/Evaluation Preds:**
```bash
python src/evaluate_test.py --checkpoint ./models/best_legend_model.pt

```



```

---

##  Citation
If you implement this persuasion detection framework or benchmark against team Legend's results, please cite our official ArabicNLP 2023 paper:

```bibtex
@inproceedings{ojo-etal-2023-legend,
    title = "Legend at {A}r{AIE}val Shared Task: Persuasion Technique Detection using a Language-Agnostic Text Representation Model",
    author = "Ojo, Olumide and Adebanji, Olaronke and Calvo, Hiram and Dieke, Damian and Ojo, Olumuyiwa and Akinsanya, Seye and Abiola, Tolulope and Feldman, Anna",
    booktitle = "Proceedings of the Arabic Language Processing Conference (ArabicNLP 2023)",
    month = dec,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    pages = "594--599",
    url = "https://aclanthology.org/2023.arabicnlp-1.61"
}

```

---
