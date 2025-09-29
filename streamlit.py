import streamlit as st
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

pinecone_api_key = os.getenv("PINECONE_API_KEY")
euron_api_key = os.getenv("EURON_API_KEY")

if not pinecone_api_key or not euron_api_key:
    st.error("Missing keys in environment")
    st.stop()


# Streamlit UI
st.title("Shanin Chatbot")
st.write("Ask questions about me!")

# Initialize session state for RAG components
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.embeddings = None
    st.session_state.retriever = None

def initialize_rag():
    try:
        if st.session_state.rag_chain is None:
            st.session_state.embeddings = download_hugging_face_embeddings()
            index_name = "portfolio"
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=st.session_state.embeddings
            )
            st.session_state.retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            chatModel = EuronChatModel()
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
            st.session_state.rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)
    except Exception as e:
        st.error(f"Error initializing RAG: {str(e)}")
        st.stop()

# Input form
with st.form("chat_form"):
    user_input = st.text_input("Your question:", placeholder="What is in my portfolio?")
    submit_button = st.form_submit_button("Ask")

# Process input
if submit_button and user_input:
    if st.session_state.rag_chain is None:
        with st.spinner("Initializing RAG pipeline..."):
            initialize_rag()
    try:
        with st.spinner("Generating response..."):
            response = st.session_state.rag_chain.invoke({"input": user_input})
            st.write("**Response:**")
            st.write(response["answer"])
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
 
