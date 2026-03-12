import streamlit as st
import pandas as pd
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.schema import Document
from langchain_classic.chains import RetrievalQA
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ API KEY not found. Please set environment variable.")
    st.stop()

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ------------------------------------------
# 1️. LOGIN SYSTEM        # In a production app, use a secure authentication method instead of hardcoded credentials.         
# ------------------------------------------

def login():
    st.title("🔐 Secure Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["authenticated"] = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# ----------------------------------------------
# 2️. LOAD DATA
# ----------------------------------------------

st.title("📰 Event-Based News RAG Chatbot")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@st.cache_resource
def load_vectorstore():

    df = pd.read_csv("data/final_event_summaries.csv")

    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore, df

vectorstore, df = load_vectorstore()
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# ---------------------------------------------
# 3️. EVENT OVERVIEW SECTION
# ---------------------------------------------

if st.sidebar:
    st.sidebar.title("🔎 Event Explorer")
    unique_events = sorted(df["event_label"].dropna().unique())

    selected_event = st.sidebar.selectbox(
        "Browse Events",
        unique_events
    )

    if selected_event:
        event_data = df[df["event_label"] == selected_event]

        st.sidebar.write("**Latest Summary:**")
        st.sidebar.write(event_data.iloc[0]["monthly_summary"])  # Display the most recent summary for the selected event

# --------------------------------------
# 4. CHAT INTERFACE
# --------------------------------------

st.subheader("💬 Ask About an Event")

query = st.text_input("Type your question here:")
if query:
    with st.spinner("Generating answer..."):
        response = qa_chain.invoke(query)
        st.success("Answer:")
        st.write(response["result"])
        st.write("#### Please give your valuable feedback on the sidebar 😊")
        st.write("#### THANK YOU! & WELCOME AGAIN! 🙏")


# -------------------------------
# 5. USER FEEDBACK SECTION
# -------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader(" 🗣 Reader Feedback")

st.sidebar.write(
"Share your thoughts about this news assistant. \
Your feedback helps us to improve the quality of information."
)

rating = st.sidebar.radio(
    "Rate this App",
    ["1 ⭐", "2 ⭐⭐", "3 ⭐⭐⭐", "4 ⭐⭐⭐⭐", "5 ⭐⭐⭐⭐⭐"],
    horizontal=True
)

feedback = st.sidebar.text_area(
"Your suggestions",
placeholder="Example: Add timeline view, improve answers, etc..."
)

if st.sidebar.button("Submit Feedback"):
    
    feedback_data = pd.DataFrame({
        "rating":[rating],
        "feedback":[feedback],
        "timestamp":[datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })

    feedback_data.to_csv(                     # Append feedback to CSV file, create if it doesn't exist
        "feedback.csv",
        mode="a",
        header=not os.path.exists("feedback.csv"),
        index=False
    )

    st.sidebar.success("✅ Thank you for your feedback!")

