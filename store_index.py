from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

# Load environment variables from .env file
load_dotenv()

# Retrieve Pinecone API key from environment variables
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Set up index
index_name = "portfolio"
index = pc.Index(index_name)

# Load and process documents
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Create Pinecone vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)