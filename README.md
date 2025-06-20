# 🧠 NLP Language Modeling – Exercise 1

This repository contains the implementation of **Exercise 1** from the Natural Language Processing course at the Hebrew University.

---

## 📚 Overview

The goal of this exercise is to build **Unigram** and **Bigram** language models based on a given corpus.  
We compute word probabilities, sentence perplexities, and complete sentences using statistical modeling.

We use **Linear Interpolation** to combine unigram and bigram models for more accurate predictions.

---

## 🧪 Features

- Train unigram and bigram language models on large text corpora
- Calculate word and sentence probabilities
- Compute perplexity for given sentences
- Perform sentence completion using probabilistic models
- Use linear interpolation:  
  `P(wᵢ|wᵢ₋₁) = λ₁ · P_unigram(wᵢ) + λ₂ · P_bigram(wᵢ|wᵢ₋₁)`

---

## 🛠️ Technologies Used

- Python 3.x
- `spaCy` for tokenization and preprocessing
- `NumPy` for vectorized probability calculations
- WikiText dataset (as `.parquet`)

---

## 📁 Files Included

| File                       | Description                                         |
|----------------------------|-----------------------------------------------------|
| `ex1.py`                  | Main script with implementation of the models       |
| `Ex1 - NLP.pdf`           | Assignment instructions (ignored by `.gitignore`)   |
| `Practical Part.pdf/docx` | Output and explanation (ignored by `.gitignore`)    |
| `.gitignore`              | Ignores large files like `.zip`, `.pdf`, `.parquet` |

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install spacy numpy
   python -m spacy download en_core_web_sm
