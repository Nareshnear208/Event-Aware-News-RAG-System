# 📰 AI-Powered News Event RAG Chatbot
## Project Overview

- This project builds an AI-powered Question & Answer chatbot that allows users to ask questions about important global news events.

- The system processes a large news dataset, identifies major events using clustering, generates monthly summaries using LLMs, and enables retrieval-based question answering (RAG).

#### Users can ask questions like:

What happened during the COVID-19 pandemic in March 2020?

What were the major economic events in 2019?

The chatbot retrieves the most relevant event summaries and generates an accurate answer using a Large Language Model.

## Project Objectives

- Transform large news datasets into structured event knowledge

- Automatically identify major events using clustering

- Generate monthly event summaries using LLMs

- Build a Retrieval Augmented Generation (RAG) chatbot

- Deploy the system for real-time question answering

## Dataset

**Dataset Used: All The News Dataset**

- ~2.6 million news articles , Multiple publishers

- Covers 2016–2020 global events

- Due to computational constraints, 25,000 articles were sampled to build the prototype pipeline.

#### Why 25K Sample?

- Enables faster processing

- Suitable for clustering experiments

- Demonstrates the full AI pipeline efficiently

- Keeps computation manageable in Kaggle environment

## Project Pipeline
### 1️. Data Cleaning & Preprocessing

- Removed null values

- Cleaned titles and article text

- Removed stopwords

- Created combined text fields for analysis

##### Output:

- clean_title
- clean_article
- text_for_embedding

### 2️. Text embeddings
- SentenceTransformer("all-MiniLM-L6-v2")

- Cosine similarity measures the angle between two vectors

- Visualize embeddings using PCA

### 3. Clustering

UMAP:

- Preserves semantic neighborhoods

- Removes noise dimensions

HDBSCAN:

- Finds dense semantic events

- Discards unrelated articles

##### output: lables id's

## 4. Cluster Labeling (Event Labels)

To identify major news topics

Features used:

TF-IDF vectors

Purpose:

Group similar articles together

Automatically detect major events

Example clusters: COVID-19 Pandemic , Global Economy , US Politics etc..


Clusters were labeled using TF-IDF keyword extraction.

Example:

Cluster keywords:

ecb, euro, bank, growth, euro zone

Event Label:

European Central Bank & Eurozone Economy

This converts machine clusters into human-readable events.

## 5. Timeline Creation

Each article contains a publication date.

- We created: year , month , year_month

- This allowed us to build a timeline of events.

##### Example:

- Event   Year	Month --> COVID-19Pandemic	   2020	3

## 6. Monthly Event Aggregation

Articles were grouped by:

- event_label + year + month

For each event-month:

- article titles collected

- article count calculated

This creates structured event summaries.

##### Example:
Event	Year	Month	Article Count ----> COVID-19 Pandemic	2020	March	  218

## 6️. LLM-Based Event Summarization

Using Groq LLM API, monthly summaries were generated.

Model used:

llama-3.1-8b-instant

Each event-month was summarized into 3–5 lines describing what happened during that period.

Example summary:

In March 2020, the COVID-19 pandemic rapidly spread across multiple countries. Governments introduced lockdown measures while global markets experienced major disruptions. Sporting events and travel were suspended worldwide.

These summaries form the knowledge base for the chatbot.

## 7️. RAG (Retrieval Augmented Generation)

The chatbot uses a RAG pipeline.

#### Steps:

- Load event summaries

- Convert summaries into embeddings

- Store embeddings in vector database

- Retrieve relevant summaries for user queries

- Generate final answer using LLM

#### This approach ensures:

- factual responses

- context-aware answers

- reduced hallucinations

#### Technologies Used
- Data Science

- Python

- Pandas

- NumPy

- Scikit-learn

- NLP

- UMAP + HDBSCAN

- TF-IDF Vectorization

- Text Cleaning

- Keyword Extraction

- LLM / AI

- Groq API

- LLaMA-3 Models

- LangChain

- RAG Components

- Text Splitting

- Embeddings

- Vector Retrieval

- Development

- Kaggle Notebook

- VS Code

- Deployment

- Git CI/CD Pipeline

- AWS EC2

- Nginx

- Streamlit

## Project Architecture
News Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Embeddings
      │
      ▼
HDBSCAN Clustering                  
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Event Labeling
      │
      ▼
Monthly Event Aggregation
      │
      ▼
LLM Summarization (Groq)
      │
      ▼
Event Knowledge Base
      │
      ▼
Embeddings + Vector Store
      │
      ▼
RAG Q&A Chatbot 

## Example Questions

Users can ask questions like:

"What happened during the COVID-19 pandemic in March 2020?"

"Major economic news in 2019"

"What global events happened in 2017?"

The chatbot retrieves relevant event summaries and generates a final response.

## Interactive chatbot UI
### 1. Application Preview
Below are some screenshots of the Event-Based News RAG Chatbot interface.

🔐 Login Page

Users must log in before accessing the application.
![alt text](<Screenshot 2026-03-11 125414-1.png>)

### 2. 📰 Main Chat Interface

Users can ask questions about real-world news events and receive AI-generated answers.
![alt text](<Screenshot 2026-03-11 114038.png>)

### 3. 📅 Event Explorer (Sidebar)
The sidebar displays all available news event categories detected from the dataset.

Users can browse events before asking questions.

![alt text](<Screenshot 2026-03-11 130358.png>)

### 4. 🗣 Reader Feedback Section

Readers can provide valuable feedback and rate the application to help improve the system.

Features include:

- Star rating system

- Suggestions box

- Feedback submission

![alt text](<Screenshot 2026-03-11 120634-1.png>)

## Doker deployments
1.Rebuild image: 
docker build -t news_rag_app

2.Run Container:
docker run -p 8501:8501 --env-file .env news_rag_app
--env-file .env ---> Passkey Groq API Key from .env 

http://localhost:8501

3.convert the local image name to Docker Hub format:
docker tag news_rag_app nareshbabu2026ai/news-rag-chatbot-streamlit:latest

4.Push to Docker Hub:
docker push nareshbabu2026ai/news-rag-chatbot-streamlit:latest

5.Test Pull From Docker Hub:(For running app from Docker Hub)
docker pull nareshbabu2026ai/news-rag-chatbot-streamlit:latest

6.Run:
docker run -p 8501:8501 --env-file .env nareshbabu2026ai/news-rag-chatbot-streamlit:latest

http://localhost:8501

## Future Improvement

Planned enhancements:

Deploy on AWS EC2

Real-time news ingestion

Larger dataset processing

Advanced vector databases (Pinecone)

## Key Skills Demonstrated

- Natural Language Processing

- Unsupervised Learning

- Event Detection

- LLM Integration

- Retrieval Augmented Generation (RAG)

- API Integration

- End-to-End AI Pipeline

## 👨‍💻 Author

Naresh Babu

AI / Data Science Enthusiast

Focused on building real-world machine learning and LLM applications.

⭐ If you found this project useful

Please consider starring the repository.
