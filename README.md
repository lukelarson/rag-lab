# RAG Q&A Application

A simple Retrieval-Augmented Generation (RAG) Question & Answering application built with LangChain and OpenAI. This application allows you to ask questions about your local documents and get AI-powered answers based on the content.

## Features

- Loads and processes local text and markdown documents
- Splits documents into chunks for efficient processing
- Uses OpenAI's text-embedding-3-small for embeddings
- Stores vectors in FAISS for fast similarity search
- Answers questions using GPT-4 with context from your documents
- Interactive command-line interface
- Persists vector store for faster subsequent runs

## Setup

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

1. Place your text (.txt) or markdown (.md) files in the `docs/` directory
2. Run the application:
   ```bash
   python rag_qa.py
   ```
3. The first run will:
   - Load your documents
   - Split them into chunks
   - Create embeddings
   - Store them in a FAISS index
4. Subsequent runs will load the existing FAISS index
5. Type your questions at the prompt
6. Type 'quit' to exit

## Project Structure

```
.
├── .env                 # Your OpenAI API key (create this file)
├── requirements.txt     # Python dependencies
├── rag_qa.py           # Main application script
├── docs/               # Place your documents here
└── faiss_index/        # Created automatically to store vectors
```

## Notes

- The application uses GPT-4 for answering questions
- Documents are split into chunks of 1000 characters with 200 character overlap
- The system retrieves the 3 most relevant chunks for each question
- Answers are limited to 3 sentences for conciseness
- Source documents are displayed for each answer

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls 