import os
from dotenv import load_dotenv
import streamlit as st
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from typing import List, Any, Optional, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.bigquery_vector_search import BigQueryVectorSearch

# Load environment variables from .env file
load_dotenv()

# Configuration
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
DATASET = os.getenv('DATASET')
TABLE = os.getenv('TABLE')
API_KEY = os.getenv('API_KEY')
# Initialize BigQuery client
client = bigquery.Client()

# Load language model
LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=4000,
    timeout=None,
    max_retries=2,
    google_api_key=API_KEY
)

class Properties:
    def __init__(self, project_id, region, dataset, table):
        self.project_id = project_id
        self.region = region
        self.dataset = dataset
        self.table = table

class BigQueryVectorSearchLocal(BigQueryVectorSearch):
    def __init__(self, project_id, dataset_name, table_name, location, embeddings, distance_strategy):
        super().__init__(
            project_id=project_id,
            dataset_name=dataset_name,
            table_name=table_name,
            location=location,
            embedding=embeddings,
            distance_strategy=distance_strategy
        )
        self.bq_client = bigquery.Client(project=project_id, location=location)

    def similarity_search(self, embedding: List[float], filter: Optional[Dict[str, Any]] = None, k: int = 4):
        # Prepare embedding list in the required format for BigQuery
        embedding_str = ', '.join(map(str, embedding))
        embedding_list = f"[{embedding_str}]"
        
        filter_expr = "TRUE"
        if filter:
            filter_expressions = [f"JSON_VALUE(base.metadata, '$.{key}') = '{value}'" for key, value in filter.items()]
            filter_expr = " AND ".join(filter_expressions)
        
        query_str = f"""
        SELECT
            base.*,
            distance AS _vector_search_distance
        FROM VECTOR_SEARCH(
            TABLE `{self.project_id}.{self.dataset_name}.{self.table_name}`,
            "ml_generate_embedding_result",
            (SELECT {embedding_list} AS embedding),
            distance_type => "COSINE",
            top_k => {k}
        )
        WHERE {filter_expr}
        LIMIT {k}
        """
        # print(query_str)
        query_job = self.bq_client.query(query_str)
        results = query_job.result()

        documents = []
        for row in results:
            metadata = {
                "industry": row["industry"],
                "rating": row["rating"],
                "raw_content": row["raw_content"],
            }
            doc = Document(page_content=row["content"], metadata=metadata)
            documents.append(doc)

        return documents

class VectorStoreFactory:
    def __init__(self, properties: Properties):
        self.embeddings = VertexAIEmbeddings(
            model_name="textembedding-gecko@003", project=properties.project_id
        )
        self.properties = properties

    def create_store(self):
        store = BigQueryVectorSearchLocal(
            project_id=self.properties.project_id,
            dataset_name=self.properties.dataset,
            table_name=self.properties.table,
            location=self.properties.region,
            embeddings=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        return store

class CustomRetriever(BaseRetriever):
    store: Any
    embeddings: Any

    def __init__(self, store, embeddings):
        super().__init__()
        self.store = store
        self.embeddings = embeddings

    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        embedding = self.embeddings.embed_query(query)
        return self.store.similarity_search(embedding=embedding, k=k)

    def with_config(self, **kwargs):
        return self

def search_by_text(query, filter=None, k=4):
    properties = Properties(PROJECT_ID, REGION, DATASET, TABLE)
    store_factory = VectorStoreFactory(properties)
    store = store_factory.create_store()
    embeddings = store_factory.embeddings
    retriever = CustomRetriever(store, embeddings)
    return retriever.get_relevant_documents(query, k=k)

def rag_search(query, k=4):
    properties = Properties(PROJECT_ID, REGION, DATASET, TABLE)
    store_factory = VectorStoreFactory(properties)
    store = store_factory.create_store()
    embeddings = store_factory.embeddings
    retriever = CustomRetriever(store, embeddings)

    # Retrieve documents
    documents = retriever.get_relevant_documents(query, k=k)
    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(LLM, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})

    return response["answer"], documents

# Streamlit app
st.title("BigQuery Vector Search and RAG App")

query = st.text_input("Enter your search query:")
k = st.slider("Number of results", min_value=1, max_value=10, value=4)

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            results = search_by_text(query, None, k)
        
        st.subheader("Search Results:")
        for i, doc in enumerate(results, 1):
            st.markdown(f"**Result {i}:**")
            st.write(f"Content: {doc.page_content}")
            st.write(f"Metadata: {doc.metadata}")
            st.write("---")
    else:
        st.warning("Please enter a search query.")

if st.button("RAG Search"):
    if query:
        with st.spinner("Generating Response..."):
            response, documents = rag_search(query, k)
        
        st.subheader("RAG Response:")
        st.write(response)
        
        st.subheader("Documents Retrieved:")
        for i, doc in enumerate(documents, 1):
            st.markdown(f"**Document {i}:**")
            st.write(f"Content: {doc.page_content}")
            st.write(f"Metadata: {doc.metadata}")
            st.write("---")
    else:
        st.warning("Please enter a search query.")
