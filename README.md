# Interactive RAG-Based QA Bot
This repository contains the code for an Interactive Retrieval-Augmented Generation (RAG)-Based QA Bot, built using Streamlit, FAISS, SentenceTransformers, and LLaMA 2. The bot allows users to upload documents, retrieve relevant information from them, and generate coherent answers to user queries in real time by leveraging a combination of document embeddings and a generative language model.

## Features
Document Upload: Users can upload multiple documents in PDF format for the bot to process.
Embeddings & Retrieval: The bot uses SentenceTransformers to encode document content into vectors, stored and searched using FAISS.
Question Answering: Users can ask queries, and the system retrieves the most relevant chunks of information from uploaded documents and generates answers using the LLaMA 2 model.
Real-Time Responses: The bot provides fast and accurate answers by combining document retrieval with the generative capabilities of LLaMA 2.
## Tech Stack
Streamlit: For the web interface.
FAISS: For efficient similarity search and retrieval of document chunks.
SentenceTransformers: For embedding the documents and queries into vectors.
PyPDF2: For reading and extracting text from PDFs.
Transformers: For using the pre-trained LLaMA 2 model from Hugging Face.
BitsAndBytes: For efficient model quantization to 8-bit, speeding up inference.
