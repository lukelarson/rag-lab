# LLM & RAG FAQ

**What is RAG?**  
RAG stands for Retrieval-Augmented Generation. It combines document retrieval with language model generation to provide answers based on external context, not just the model’s training data.

**Why use embeddings?**  
Embeddings are vector representations of text that allow for semantic similarity search. In a RAG pipeline, embeddings are used to find the most relevant chunks of information to feed into the LLM.

**What’s the benefit of RAG over fine-tuning?**  
RAG is faster, cheaper, and more flexible than fine-tuning. You can update or expand your knowledge base without retraining the model.

**How many chunks should I retrieve for context?**  
It depends on your use case, but most systems retrieve 3 to 5 chunks. More chunks improve recall, but too many can cause prompt bloat and reduce generation quality.

**Does RAG eliminate hallucinations?**  
Not completely. But by grounding the model’s outputs in factual source material, RAG significantly reduces hallucination risk and improves trustworthiness.

**What’s the difference between FAISS and other vector stores?**  
FAISS is an open-source, efficient vector store developed by Facebook. It's widely used for its speed and scalability, especially in local development and prototyping.
