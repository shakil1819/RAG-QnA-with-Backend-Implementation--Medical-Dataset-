# FastAPI LLM RAG Backend Application
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest)
![FAISS](https://img.shields.io/badge/FAISS-0066CC?style=for-the-badge&logo=facebook)
![LangChain](https://img.shields.io/badge/LangChain-2563EB?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai)
![Unit Tests](https://img.shields.io/badge/Tests-Unit%20Tests-6DA55F?style=for-the-badge&logo=pytest)
![Integration Tests](https://img.shields.io/badge/Tests-Integration%20Tests-6DA55F?style=for-the-badge&logo=pytest)
![Logging](https://img.shields.io/badge/Logging-Active-4B8BBE?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python)



## Overview
This project is a FastAPI backend application that implements a Retrieval-Augmented Generation (RAG) model using a Large Language Model (LLM) on Medical Datasets (CSV). The application is designed to provide intelligent responses by combining the capabilities of LLMs with a retrieval mechanism to fetch relevant information from a knowledge base.

## Features
- FastAPI framework for building APIs quickly and efficiently.
- Integration with a Large Language Model for natural language understanding and generation.
- Retrieval mechanism to fetch relevant documents or data to enhance responses.
- Easy to deploy and scale.

## System Architecture
![Untitled diagram-2024-10-30-140232](https://github.com/user-attachments/assets/e9179dd7-5aa1-4151-89ee-083e504fedb7)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shakil1819/RAG-QnA-with-Backend-Implementation--Medical-Dataset-.git
   ```

2. Create a virtual environment inside the codebase folder:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   mkdir logs/ vectors/
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Set necessary keys in `.env`. follow `.env.sample`.
5. For the first time running , run
   ```python
   python3 app/rag/llm.py --repopulate
   ```
   This will generate and store FAISS vectore store and index
## Usage
1. Build and Run docker container
   ```
   docker compose -f docker-compose.yml up --build
   ```

3. Access the API documentation at `http://127.0.0.1:8000/docs`.

## API Endpoints
- `POST /question`: Generates a response based on the input query.
- `GET /health`: Checks the health status of the application.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

