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

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
euron_api_key = os.getenv("EURON_API_KEY")

if not pinecone_api_key or not euron_api_key:
    st.error("Missing keys in environment")
    st.stop()

# Sidebar Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["About Me", "Chatbot"])

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.embeddings = None
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

def initialize_rag():
    try:
        if st.session_state.rag_chain is None:
            st.session_state.embeddings = download_hugging_face_embeddings()
            index_name = "portfolio"
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=st.session_state.embeddings
            )
            st.session_state.retriever = docsearch.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )
            chatModel = EuronChatModel()
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
            st.session_state.rag_chain = create_retrieval_chain(
                st.session_state.retriever, question_answer_chain
            )
    except Exception as e:
        st.error(f"Error initializing RAG: {str(e)}")
        st.stop()

# About Me Section
if menu == "About Me":
    st.title("👨 About Me")
    st.image("static/shanin.jpeg", width=120)  # Replace with your image
    st.markdown("""
    ### Hi, I'm **Shanin Hossain** 👋  
    I'm an AI Engineer & Research Assistant passionate about:
    - Machine Learning, Deep Learning, and Generative AI  
    - Computer Vision & Natural Language Processing  
    - Healthcare Informatics and Medical Imaging  

    📌 I have worked on multiple AI projects, including OCR, Retrieval-Augmented Generation (RAG), and hybrid graph networks.  
    """)
    st.success("👉 Navigate to **Chatbot** in the sidebar to chat with me!")

# Chatbot Section
elif menu == "Chatbot":
    st.title("🤖 Shanin Chatbot")
    st.write("Ask me anything about my portfolio!")

    # Display previous messages (ChatGPT-style)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Input box (ChatGPT style)
    if prompt := st.chat_input("Type your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Initialize RAG if needed
        if st.session_state.rag_chain is None:
            with st.spinner("Initializing RAG pipeline..."):
                initialize_rag()

        try:
            with st.spinner("Generating response..."):
                response = st.session_state.rag_chain.invoke({"input": prompt})
                answer = response["answer"]

                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
