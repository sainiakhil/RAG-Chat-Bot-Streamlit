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



## 1. Model Architecture
- Embedding Model
SentenceTransformer: We use the SentenceTransformer model, specifically the paraphrase-MiniLM-L6-v2 for embedding the uploaded document texts and the user query. The model encodes each text segment into a vector, capturing semantic meaning and context.
Why MiniLM-L6-v2?: It’s a small but effective model for sentence embeddings, allowing for faster computation while maintaining reasonable accuracy.
-  Vector Database (FAISS)
FAISS (Facebook AI Similarity Search): This vector database is used to perform nearest-neighbor searches on the document embeddings. FAISS allows for efficient similarity searches even with a large number of embeddings.
The embeddings of document chunks are stored in FAISS, which is queried when a user asks a question to retrieve the top-k similar documents (in this case, k=3).
- Generative Model
LLaMA 2 (LLaMA-2-7B Chat): We use LLaMA 2 as the language model to generate responses based on the query and retrieved context. This model processes the user’s question alongside relevant retrieved document sections to create an accurate and contextually aware answer.

BitsAndBytes Quantization: The model is loaded in 8-bit precision using BitsAndBytesConfig, allowing for faster computation and lower memory usage while maintaining model accuracy. This allows for better efficiency when using large models like LLaMA-2.

## 2. Approach to Retrieval
- Document Upload and Chunking: When a user uploads PDF documents, the bot extracts the text from each page using PyPDF2 and chunks the text into segments of 1000 characters. This chunking improves the quality of embeddings, especially for longer documents.

- Embedding Generation: After chunking, embeddings are generated for each document chunk using SentenceTransformer, which converts each chunk into a high-dimensional vector.

- Indexing with FAISS: The document embeddings are added to a FAISS index. FAISS allows for fast similarity searches, which is crucial when processing multiple documents in real-time.

- Query Processing: When the user asks a question, it is encoded into a query vector using the same embedding model. This query vector is then compared against the FAISS index to find the top-3 most similar document chunks.

## 3. Generating Responses
- Contextual Query Augmentation: Once the relevant document chunks are retrieved from the FAISS index, the system concatenates the original question and retrieved information to form an augmented input. This augmented input includes the user’s query and context from the documents, which is then passed to the LLaMA-2 model.

- Response Generation: The augmented input is tokenized using AutoTokenizer, and passed through the AutoModelForCausalLM to generate a response. The model is set to a specific number of tokens, temperature, and top-p values to control the diversity and coherence of the output.

- Display of Final Answer: The generated response is decoded and presented to the user as the final answer. If no relevant document chunks are found (i.e., no meaningful similarity), a fallback message is shown: "No relevant document found."

## 4. How Responses are Created
The response generation follows this flow:
User inputs a query.
Query is converted into an embedding vector and searched within the FAISS index.
Top-3 relevant document chunks are retrieved.
These chunks are concatenated with the user query and passed to LLaMA-2 to generate the final answer.
The generated answer is returned and displayed to the user.
## 5. Deployment Using Ngrok
Ngrok is used to expose the Streamlit app to the public. This allows users to interact with the RAG bot from any location, without needing to set up a local server.

## 6. How to Run the Application
- Install Dependencies: Ensure all necessary packages (PyTorch, SentenceTransformer, FAISS, PyPDF2, Transformers, Streamlit) are installed.

!pip install sentence-transformers faiss-cpu PyPDF2 transformers streamlit torch

- Run the Application:

The application can be run using Streamlit.

!streamlit run app.py

- Expose the Application with Ngrok:

Ngrok is used to make the application publicly accessible.
Start Ngrok:

ngrok.set_auth_token("your-ngrok-token")
public_url = ngrok.connect(addr='8501', proto='http', bind_tls=True)

After starting Streamlit and Ngrok, the application will be live at the provided public URL.





