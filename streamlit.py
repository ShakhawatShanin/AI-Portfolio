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

# ✅ Minimal CSS – Only apply Fira Code font globally + Sidebar styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap');

/* Global font */
* {
    font-family: 'Fira Code', monospace !important;
}

/* Sidebar background & padding */
[data-testid="stSidebar"] {
    background-color: #f0f2f6 !important; /* Light ash */
    border-radius: 0 20px 20px 0;
    padding: 25px 15px;
}

/* Sidebar title */
[data-testid="stSidebar"] .css-1d391kg {
    color: #1f2937 !important;
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 25px;
}

/* Sidebar radio buttons as styled buttons */
[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
[data-testid="stSidebar"] .stRadio label {
    background-color: rgba(31,41,55,0.1) !important; /* subtle gray */
    color: #1f2937 !important;
    padding: 12px 18px;
    border-radius: 12px;
    transition: all 0.3s ease;
    cursor: pointer;
    font-size: 1.1rem;
    font-weight: 500;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(31,41,55,0.2) !important;
    transform: translateX(5px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
[data-testid="stSidebar"] .stRadio input:checked + label {
    background-color: #4B5563 !important; /* Darker for selected */
    color: #ffffff !important;
}

/* Sidebar image */
[data-testid="stSidebar"] img {
    border-radius: 15px;
    margin-bottom: 20px;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border: 1px solid rgba(0,0,0,0.2) !important;
    margin: 20px 0;
}

/* Sidebar caption */
[data-testid="stSidebar"] .css-1x8i1po {
    text-align: center;
    font-size: 0.9rem;
    color: #4b5563 !important;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Menu
with st.sidebar:
    # st.title("🌟 Navigation")
    st.image("static/icon.png", width=120)  # Image in sidebar
    st.markdown("---")
    menu = st.radio("Go to", ["About Me", "Chatbot"], label_visibility="hidden")
    st.markdown("---")
    st.caption("© 2025 Shanin Hossain")
    # st.markdown("""
    # <div style="text-align:center; margin-top: 15px;">
    #     <a href="https://www.linkedin.com/in/shanin-hossain" target="_blank">
    #         <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="25" style="margin:0;">
    #     </a>
    #     <a href="https://github.com/shaninhossain" target="_blank">
    #         <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" width="25" style="margin:0;">
    #     </a>
    #     <a href="https://www.facebook.com/shaninhossain" target="_blank">
    #         <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/facebook.svg" width="25" style="margin:0;">
    #     </a>
    # </div>
    # """, unsafe_allow_html=True)



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
# About Me Section
if menu == "About Me":
    st.write("Hey there 👋")
    st.title("👨 I'm Shanin Hossain")
    st.markdown("""
    I'm an AI Engineer & Research Assistant passionate about:
    - Machine Learning, Deep Learning, and Generative AI  
    - Computer Vision & Natural Language Processing  
    - Healthcare Informatics and Medical Imaging  
    
    📌 I have worked on multiple AI projects, including OCR, Retrieval-Augmented Generation (RAG), and hybrid graph networks.  
    """)
    st.success("👉 Navigate to **Chatbot** in the sidebar to chat with me!")
    st.header("My Projects", divider="gray")

    # --- Project Data ---
    projects = [
        {
            "name": "Brain Glioma Grading System",
            "description": "Developed a hybrid graph neural network to grade glioma tumors from medical imaging data.",
            "tech_stack": ["PyTorch Geometric", "Graph Neural Networks", "Medical Imaging", "Python"]
        },
        {
            "name": "OCR Automation",
            "description": "Built an OCR pipeline for document image understanding and text extraction.",
            "tech_stack": ["YOLOv8", "OpenCV", "Tesseract OCR", "FastAPI"]
        },
        {
            "name": "Portfolio Chatbot",
            "description": "Created a RAG-powered chatbot integrated with Pinecone & custom embeddings for Q&A over portfolio data.",
            "tech_stack": ["Streamlit", "LangChain", "Pinecone", "Hugging Face"]
        },
    ]

    # --- Render Projects with Badges ---
    for project in projects:
        with st.container():
            st.subheader(project["name"])
            st.write(project["description"])

            # Create badges for each tech in stack
            badges_html = " ".join([
                f"<span class='badge'>{tech}</span>" for tech in project["tech_stack"]
            ])

            st.markdown(
                f"<div style=''>{badges_html}</div>",
                unsafe_allow_html=True
            )

            st.markdown("---")

    # --- Badge Styling ---
    st.markdown("""
    <style>
    .badge {
        display: inline-block;
        padding: 6px 12px;
        font-size: 0.70rem;
        font-weight: 600;
        color: white;
        border-radius: 12px;
    }

    /* Randomized color palette */
    .badge:nth-child(5n+1) { background-color: #2563EB; }  /* Blue */
    .badge:nth-child(5n+2) { background-color: #059669; }  /* Green */
    .badge:nth-child(5n+3) { background-color: #D97706; }  /* Orange */
    .badge:nth-child(5n+4) { background-color: #9333EA; }  /* Purple */
    .badge:nth-child(5n+5) { background-color: #DC2626; }  /* Red */
    </style>
    """, unsafe_allow_html=True)


    st.header("Publications", divider="gray")

    # --- Publication Data ---
    publications = [
        {
            "title": "Automated Detection of Age-Related Macular Degeneration (AMD) Using Deep Learning",
            "venue": "Journal of Medical Imaging & Health Informatics, 2023",
            "description": "Published a deep learning-based pipeline to detect age-related macular degeneration from retinal images.",
            "link": "https://doi.org/xxxxxx"
        },
        {
            "title": "Using Hyperdimensional Computing to Extract Features for the Detection of Type 2 Diabetes",
            "venue": "Conference on Health Informatics, 2024 (Under Review)",
            "description": "Explored hyperdimensional computing techniques to improve detection of Type 2 Diabetes from clinical data.",
            "link": ""
        }
    ]

    # --- Render Publications ---
    for pub in publications:
        with st.container():
            st.subheader(pub["title"])
            st.caption(pub["venue"])
            st.write(pub["description"])

            if pub["link"]:
                st.markdown(f"[🔗 View Publication]({pub['link']})")

            st.markdown("---")



# Chatbot Section
elif menu == "Chatbot":
    st.title("🤖 Shanin Chatbot")
    st.write("Ask me anything about my portfolio!")

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input box
    if prompt := st.chat_input("Type your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

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
                with st.chat_message("assistant"):
                    st.write(answer)

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
