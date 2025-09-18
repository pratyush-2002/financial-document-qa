# 💰 Financial Document Q&A Assistant

A simple Streamlit app that lets you upload **financial documents** (PDF or Excel) and ask **natural-language questions** about revenue, expenses, profits, and other metrics.  
Built with **LangChain**, **Chroma**, and **Ollama** for local LLM inference.

---

## Features
- Upload PDF and Excel files.
- Extract and process financial data automatically.
- Ask natural-language questions about the uploaded files.
- Runs locally using Ollama’s Mistral model.

---

## Installation
Clone this repository and install the requirements:

```bash
git clone https://github.com/yourusername/financial-document-qa.git
cd financial-document-qa
pip install -r requirements.txt
```

## Need Ollama to pull mistral
```bash
ollama pull mistral
```
## Use streamlit for running
```
streamlit run app.py
```
