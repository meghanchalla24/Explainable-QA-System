# Explainable Question Answering System for BITS Institute Regulations

This project is an explainable question-answering (QA) system designed to help users query and understand BITS Pilani institute regulations. It uses a **hybrid approach** for document retrieval, followed by **reranking**, and finally a **generator model** to refine the retrieved answers.

##  Features

- Hybrid retrieval combining sparse and dense retrieval methods.
- 'Flashrank' model for reranking for relevance.
- Answer refinement using a large language model.
- Streamlit-based interactive interface.

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/meghanchalla24/Explainable-QA-System.git
cd Explainable-QA-System
```

### 2. Create a Virtual Environment
#### For Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### For macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```


### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


## Usage
Run the following command to launch the Streamlit interface:

python main.py

## Project Structure

- main.py                  # Entry point for launching the Streamlit app
- requirements.txt         # Required Python packages
- project_pdfs/            # Folder with source documents (not only PDFs)
- prepare_corpus.py        # Builds corpus; customizable paths and URLs
- document_statistics.py   # Corpus metadata and statistics
- document_retrieval.py    # Hybrid retrieval and reranking logic
- generator.py             # Answer generation using LLaMA model
- visual_plots.py          # Contains plots as per metrics evaluation
- boolean_retrieval.py     # Check boolean retrieval

## Corpus
By default, the corpus is built from the project_pdfs/ folder.
To use your own dataset:

Update the folder path in prepare_corpus.py.

Optionally modify the list of base URLs in the same file.

## Model Used
Generator: meta-llama/Llama-4-Scout-17B-16E-Instruct

## Author
Meghan Challa

## Acknowledgements
Streamlit for the UI framework

Together AI for model and tokenizer hosting"# Explainable-QA-System" 
