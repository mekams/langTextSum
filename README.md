# 🧠 Exploring Automatic Text Summarisation Using a Modular Pipeline Approach with LangChain

A Master's dissertation project exploring a **hybrid extractive-abstractive text summarisation system** using a **modular pipeline** built with **LangChain** and **Hugging Face** models.

---

## 📌 Project Overview

This research project proposes a **modular pipeline** for **Automatic Text Summarisation (ATS)** by combining extractive methods like **LexRank** with powerful abstractive models like **BART**. The approach aims to improve summary **coherence, accuracy, and computational efficiency** over traditional methods.

---

## 📚 Features

- 🔁 **Modular Pipeline**: Combines extractive and abstractive techniques.
- 📦 **LangChain Integration**: Enables chaining of processing steps for clean modularity.
- 🤖 **Models Used**:
  - Extractive: LexRank (via Sumy)
  - Abstractive: BART (via Hugging Face Transformers)
- 📊 **Evaluation Metrics**: ROUGE-1, ROUGE-2, ROUGE-L
- 📈 **Visualization**: Word clouds and bar charts for performance insights.
- 🌐 **Streamlit App**: Interactive web interface for summarising text or `.docx` files.

---

## 🧪 Implementation Highlights

### 🧰 Tools & Frameworks

- **LangChain**: Modular NLP chaining
- **Hugging Face Transformers**: BART & T5 models
- **Sumy**: Extractive summarisation via LexRank
- **Streamlit**: Web interface for summarisation
- **Google Colab**: Model testing and evaluation
- **ROUGE**: Summary evaluation metrics

### ⚙️ Pipeline Flow

```text
Input Text/Document
   ↓
Text Splitting (LangChain)
   ↓
LexRank (Extractive Summarisation)
   ↓
BART (Abstractive Summarisation)
   ↓
ROUGE Evaluation + Word Clouds

# Deployment 
https://langtextsum.streamlit.app/
