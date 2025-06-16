"""
RAG Q&A Application using LangChain and OpenAI.

This script implements a simple RAG (Retrieval-Augmented Generation) system that:
1. Loads documents from a local directory
2. Processes and chunks the documents
3. Creates embeddings using OpenAI's text-embedding-3-small
4. Stores vectors in FAISS
5. Answers questions using GPT-4 and the retrieved context
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def load_documents(directory: str) -> List:
    """
    Load documents from the specified directory.
    Supports both .txt and .md files.
    """
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            try:
                if ext == ".txt":
                    loader = TextLoader(file_path)
                elif ext == ".md":
                    loader = UnstructuredMarkdownLoader(file_path)
                else:
                    print(f"Skipping unsupported file type: {file_path}")
                    continue
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
    return documents

def create_chunks(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into chunks suitable for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    return text_splitter.split_documents(documents)

def create_vector_store(chunks: List):
    """
    Create and return a FAISS vector store from document chunks.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create and persist the vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def setup_qa_chain(vector_store):
    """
    Set up the RetrievalQA chain with a custom prompt.
    """
    # Custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, say "I don't know". Do NOT try to make up an answer.
    Use three sentences maximum and keep the answer concise and factual.

    Context: {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize the LLM
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set your OpenAI API key in the .env file")

    # Create docs directory if it doesn't exist
    os.makedirs("docs", exist_ok=True)
    
    # Check if vector store exists
    if os.path.exists("faiss_index"):
        print("Loading existing vector store...")
        embeddings = OpenAIEmbeddings(
             model="text-embedding-3-small",
             openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        try:
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading vector store: {e}")
            print("Creating new vector store...")
            documents = load_documents("docs")
            chunks = create_chunks(documents)
            vector_store = create_vector_store(chunks)
            vector_store.save_local("faiss_index")
    else:
        print("Loading documents...")
        documents = load_documents("docs")
        print("Creating chunks...")
        chunks = create_chunks(documents)
        print("Creating vector store...")
        embeddings = OpenAIEmbeddings(
             model="text-embedding-3-small",
             openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        vector_store = create_vector_store(chunks)
        vector_store.save_local("faiss_index")
    
    # Set up QA chain
    qa_chain = setup_qa_chain(vector_store)
    
    # Interactive Q&A loop
    print("\nRAG Q&A System Ready! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'quit':
            break
            
        if not question:
            continue
            
        try:
            # Get answer
            result = qa_chain({"query": question})
            
            # Print answer
            print("\nAnswer:", result["result"])
            
            # Print sources if available
            if result.get("source_documents"):
                print("\nSources:")
                for doc in result["source_documents"]:
                    print(f"- {doc.metadata.get('source', 'Unknown source')}")
                    
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 