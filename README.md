# BigQuery Vector Search and RAG App

This project is a Streamlit application that allows users to perform vector searches on a BigQuery dataset and retrieve relevant documents using a Retrieval-Augmented Generation (RAG) approach. The app integrates Google Vertex AI for embeddings and LangChain for chaining document retrieval and language generation tasks.

## Features

- **Vector Search**: Perform a vector search on a BigQuery dataset to retrieve relevant documents.
- **RAG Search**: Generate a response to a query based on the retrieved documents using a RAG approach.
- **Configuration via `.env`**: Securely configure project settings and API keys using environment variables.

## Prerequisites

- Python 3.7+
- Google Cloud Project with BigQuery and Vertex AI enabled
- BigQuery dataset with embedded vectors

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jason202005/rag-gcp-streamlit.git
   cd rag-gcp-streamlit
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with the following content:
   ```env
   PROJECT_ID=your-google-cloud-project-id
   REGION=your-region
   DATASET=your-dataset
   TABLE=your-table
   API_KEY=your-google-api-key
   ```

   Environment Variables

- `PROJECT_ID`: Your Google Cloud project ID.
- `REGION`: The region where your BigQuery dataset is located.
- `DATASET`: The BigQuery dataset name.
- `TABLE`: The BigQuery table name.
- `API_KEY`: Your Google API key for accessing Vertex AI and other Google Cloud services.

   


5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Open the Streamlit app**
   - Navigate to `http://localhost:8501` in your web browser.

2. **Perform a Vector Search**
   - Enter your search query in the "Enter your search query" input box.
   - Use the slider to select the number of results to retrieve.
   - Click on the "Search" button to perform the vector search.

3. **Perform a RAG Search**
   - Enter your search query in the "Enter your search query" input box.
   - Use the slider to select the number of results to retrieve.
   - Click on the "RAG Search" button to generate a response based on the retrieved documents.

## Project Structure

```
bigquery-vector-search-rag/
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables file
├── requirements.txt       # Python dependencies
└── README.md              # Project README
```

## Dependencies

- `streamlit`: Web application framework for creating interactive apps.
- `google-cloud-bigquery`: Client library for interacting with BigQuery.
- `langchain-google-vertexai`: LangChain integration for Google Vertex AI.
- `langchain-google-genai`: LangChain integration for Google Generative AI.
- `python-dotenv`: Library for loading environment variables from a `.env` file.


## References
https://cloud.google.com/vertex-ai/docs/vector-search/quickstart https://github.com/rocketechgroup/langchain_bigquery_vector/tree/main
