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
Additionally, run the following command to download the lightweight reranker model locally:
```bash
pip install "rerankers[flashrank]"
```


## Usage
Run the following commands after activating your virtual environment to launch the Streamlit interface:
```bash
python main.py
```
```bash
streamlit run main.py
```

## Project Structure
### main.py
This is the entry point to the application. The file provides options (tabs) to be displayed when hosted via a streamlit interface. One tab is for viewing the statistics of the corpus before and after preprocessing, one tab for viewing the plots of various scores so as to evaluate application performance, one tab for querying. 

### prepare_corpus.py and preprocess_documents.py
These files are specifically for preparing the corpus. If you wish to create a different corpus or make any changes to the original documents (documents used other than the actual uploaded documents), execute the following code snippet (make sure the file in which you run the code is in src):
```bash
from prepare_corpus import process_files
from preprocess_documents import data_clean,spell_correction

pdf_folder = r"D:\Information_retrieval_project\project_pdfs" # replace with the actual location of your document folder
initial_preprocessed_files = process_files(pdf_folder) # this will create a json file with initial extracted details from documents
final_corrected_extracted_data = spell_correction(initial_preprocessed_files) # this creates a json file after applying spelling correction
cleaned_extracted_data = data_clean(final_corrected_extracted_data) # this creates a json file after data cleaning and normalization
```
Now we have all our json files ready which we would be using in further retrieval tasks. A sample structure has been mention in data/processed, where there are three json files used already for the project. Also update the url links in the 'prepare_corpus.py file', where there is a dictionary defined for custom urls. 

### documents_statistics.py
This file contains function for calculating the statistics of the coprus. This function has been invoked in the main.py file under the "View Statistics" tab.

### document_retrieval.py
This file contains the functions for performing hybrid retrieval and reranking. This wont be used explicitly unless you want to modify the functions as per your need. 

### generator.py 
This file is for getting responses from the language model (meta-llama/Llama-4-Scout-17B-16E-Instruct) from the retrieved text

### visual_plots.py
This file is used for evaluating the metrics and for respective plotting of these metrics

**Note** : Try to use absolute paths over relative paths wherever necessary


## Author
Meghan Challa

## Acknowledgements
Streamlit for the UI framework

Together AI for model and tokenizer hosting"# Explainable-QA-System" 
