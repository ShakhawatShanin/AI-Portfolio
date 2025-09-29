from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from src.euron_chat import EuronChatModel
from dotenv import load_dotenv
import os

# Load environment variables from .env file (for local dev)
load_dotenv()

# Verify API keys (optional: remove in full production if confident)
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
euron_api_key = os.environ.get("EURON_API_KEY")
if not pinecone_api_key or not euron_api_key:
    raise ValueError("Missing required API keys in environment variables.")

app = Flask(__name__)

# -----------------------------
# Load embeddings and Pinecone index
# -----------------------------
embeddings = download_hugging_face_embeddings()
index_name = "portfolio"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -----------------------------
# Initialize EuronChatModel & RAG chain
# -----------------------------
chatModel = EuronChatModel()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])

    return str(response["answer"])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)  # Debug=False for production