Below is a sample **README.md** file that explains each part of the code, how to set up the environment, and how to run the project. Feel free to modify sections (like the license or your personal details) as you see fit.

---

# RAG Pipeline Demo with MongoDB Atlas Vector Search & OpenAI

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using:

- **Dummy Resume PDFs** for data ingestion
- **SentenceTransformer** for embedding textual data
- **MongoDB Atlas Vector Search** for storing embeddings and performing **vector similarity search**
- **OpenAI** for generating an AI-powered summary or recommendation

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Project Structure](#project-structure)  
5. [Setup & Installation](#setup--installation)  
6. [Usage](#usage)  
   1. [Generating Dummy PDFs](#1-generate-dummy-pdfs)  
   2. [Embedding & Storing in MongoDB](#2-embed--store-in-mongodb)  
   3. [Vector Search](#3-run-a-query--vector-search)  
   4. [Optional: Summarize with OpenAI](#4-optional-summarize-with-openai)  
7. [Environment Variables](#environment-variables)  
8. [Detailed Code Walkthrough](#detailed-code-walkthrough)  
   1. [Dummy PDF Generation](#dummy-pdf-generation)  
   2. [RAGPipeline Class](#ragpipeline-class)  
      - [Initialization](#initialization)  
      - [read_and_chunk_pdf](#read_and_chunk_pdf)  
      - [embed_and_store](#embed_and_store)  
      - [search](#search)  
      - [ask_chatgpt](#ask_chatgpt)  
9. [Known Issues / Troubleshooting](#known-issues--troubleshooting)  
10. [License](#license)  

---

## Overview

This project is a minimal “prototype” for a RAG pipeline:

1. It generates **dummy resumes** in PDF format for various job roles (like plumber, electrician, personal trainer, etc.).  
2. It chunks and **embeds** the PDF text using [SentenceTransformers](https://www.sbert.net/).  
3. It stores the embeddings in **MongoDB Atlas** using the new **Vector Search** feature.  
4. When a user query is provided, it performs a **vector similarity search** in MongoDB.  
5. Lastly, it optionally calls the **OpenAI** API to summarize or recommend a candidate based on the retrieved results.

---

## Features

- **Automatic PDF Generation**: Creates 2–3 random resumes for each job role with semi-realistic text.  
- **Chunking & Embedding**: Splits PDF text into ~150-word chunks, then embeds them with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).  
- **MongoDB Atlas Integration**: Stores embeddings in a collection with a **Vector Search** index for fast similarity queries.  
- **RAG Summaries**: Uses OpenAI’s chat completions to produce an answer or recommendation from the retrieved chunks.  
- **Environment-based Config**: Credentials (MongoDB URI, OpenAI API Key) are loaded from environment variables.

---

## Tech Stack

- **Python 3.8+**
- **[PyMuPDF (fitz)](https://pypi.org/project/PyMuPDF/)** for reading PDF text
- **[FPDF](https://pypi.org/project/fpdf/)** for creating PDF files
- **[Sentence Transformers](https://pypi.org/project/sentence-transformers/)** for embedding
- **[PyMongo](https://pypi.org/project/pymongo/)** for MongoDB integration
- **[OpenAI Python Library (>=1.0.0)](https://pypi.org/project/openai/)** for chat completions
- **[tqdm](https://pypi.org/project/tqdm/)** for progress bars
- **[Python-dotenv](https://pypi.org/project/python-dotenv/)** for environment variable management

---

## Project Structure

```
.
├── .env               # Contains environment variables like OPENAI_API_KEY, MONGO_URI
├── dummy_resumes/     # PDF output folder automatically created by the script
├── rag_pipeline.py    # Main script containing the pipeline logic
├── requirements.txt   # Required Python dependencies
└── README.md          # This README file
```

---

## Setup & Installation

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a Virtual Environment** (recommended):  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   # or:
   # venv\Scripts\activate.bat # On Windows
   ```

3. **Install Dependencies**:  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Add Environment Variables**:  
   - Copy `.env.example` to `.env` (if you have such a file) or create a new `.env`.
   - Ensure it contains your **MongoDB** connection string and **OpenAI API** key:
     ```
     MONGO_URI=mongodb+srv://USERNAME:PASSWORD@yourcluster.mongodb.net/?retryWrites=true&w=majority
     OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ```
   - **Never** commit your real API keys to source control.

---

## Usage

### 1) Generate Dummy PDFs

By default, the script can generate 2-3 PDF resumes for each job role. Uncomment the lines in `if __name__ == "__main__":`:

```python
dummy_pdf_paths = generate_dummy_pdfs()
print(f"Generated {len(dummy_pdf_paths)} dummy PDF resumes in the 'dummy_resumes' folder.")
```

This will create PDFs in the `dummy_resumes/` folder.

### 2) Embed & Store in MongoDB

After generating the PDFs, embed them and insert into MongoDB:
```python
pipeline.embed_and_store(dummy_pdf_paths)
```
This calls:
- `read_and_chunk_pdf()` to extract text in 150-word chunks
- `self.embedding_model.encode()` to create embeddings
- `insert_many()` to store them in your Atlas cluster

### 3) Run a Query / Vector Search

```python
test_query = "I need a personal trainer to help me stay fit."
results = pipeline.search(test_query, top_k=3)
```
This will:
- Embed the query
- Perform `"$vectorSearch"` with `"queryVector"` on the stored embeddings
- Return the top 3 matches

### 4) (Optional) Summarize with OpenAI

Use the retrieved docs and pass them into the `ask_chatgpt()` method:
```python
openai_api_key = os.getenv("OPENAI_API_KEY")
final_answer = pipeline.ask_chatgpt(test_query, results, openai_api_key)
print("ChatGPT Answer:\n", final_answer)
```
This calls the **OpenAI** API to produce a final answer (e.g. recommendation or summary).

---

## Environment Variables

- **MONGO_URI**: Your MongoDB Atlas connection string. Must have the Atlas Vector Search Beta or GA features enabled.  
- **OPENAI_API_KEY**: Your OpenAI API key. Must be a valid key to call the OpenAI Chat endpoints.

Example `.env`:
```
MONGO_URI="mongodb+srv://<user>:<password>@cluster0.abcd123.mongodb.net/?retryWrites=true&w=majority"
OPENAI_API_KEY="sk-..."
```

---

## Detailed Code Walkthrough

### Dummy PDF Generation

1. **Random Resume Fields**: The functions `random_person_name()`, `random_phone()`, `random_email()`, etc., produce semi-realistic but random personal data.  
2. **create_pdf()**: Uses the `FPDF` library to create a single PDF resume file containing the generated text (name, phone, email, objective, skills, experience).  
3. **generate_dummy_pdfs()**: Loops over predefined job titles and creates 2-3 PDF files each.

### RAGPipeline Class

This class wraps the ingestion, embedding, storage, search, and summarization logic.

#### Initialization

- Connects to MongoDB Atlas via the `MongoClient` using `server_api=ServerApi('1')`.
- Pings the cluster to confirm connectivity.
- Instantiates a `SentenceTransformer` model (`all-MiniLM-L6-v2`) for embeddings.

#### `read_and_chunk_pdf(pdf_path, words_per_chunk=150)`

- Opens a PDF using **PyMuPDF** (`fitz.open`).
- Extracts text, splits it into ~150-word chunks.
- Returns a list of chunk strings.

#### `embed_and_store(pdf_files)`

- Reads each file → splits into chunks → encodes each chunk with SentenceTransformer.
- Converts the embedding to a Python list and stores it in MongoDB, along with metadata (e.g., `_id`, file name, chunk text).
- Clears the collection first (`delete_many({})`) to avoid duplicates from previous runs.

#### `search(query, top_k=3)`

- Embeds the user query, then runs a `$vectorSearch` aggregation pipeline in MongoDB:
  ```json
  {
    "$vectorSearch": {
      "index": self.index_name,
      "queryVector": query_emb,
      "path": "embedding",
      "limit": top_k,
      "numCandidates": 50
    }
  }
  ```
- Returns top-k matching chunks, each with a `score` for similarity.

#### `ask_chatgpt(query, retrieved_docs, openai_api_key)`

- Uses `openai.Client(...)` to call the **OpenAI** Chat Completions API (`model="gpt-4o-mini"`, or your choice).
- Builds a context by concatenating chunk text.  
- Sends a system prompt + user prompt, and returns the model’s completion text.

---

## Known Issues / Troubleshooting

- **PyMuPDF (fitz) Version**: Make sure your version of `PyMuPDF` is 1.18.14+ to avoid text extraction issues.  
- **OpenAI `APIRemovedInV1`**: If you see this error, ensure you’re using `openai>=1.0.0` and not calling the old `openai.ChatCompletion.create()`. The correct usage is `openai.chat.completions.create(...)`.  
- **MongoDB Vector Search**: You must create a “Lucene Vector” index in your Atlas cluster (version 8.0.4+). Check that your index mapping is correct:
  ```json
  {
    "mappings": {
      "dynamic": false,
      "fields": {
        "embedding": {
          "type": "vector",
          "dimensions": 384,
          "similarity": "cosine"
        }
      }
    }
  }
  ```
- **Index Name**: If your index name is not `"default"`, you must update `index_name` accordingly in the `RAGPipeline` constructor.
- **Environment**: If your `.env` is not being loaded, ensure you installed `python-dotenv` and have `load_dotenv()` at the top of your script.

---

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). You are free to modify and use the code for your own purposes. Attribution is appreciated but not required.

---

**Enjoy exploring the code and building your own RAG pipeline with MongoDB + OpenAI!** If you have any questions or issues, feel free to open an issue or submit a pull request. Good luck!
