import streamlit as st
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


st.title("📚 PDF Query Bot")
st.markdown("Ask questions about **FidelAI**, a platform built as part of an undergraduate senior research project. Answers are based entirely on the project's documentation and research paper.")


@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="pdf_chatbot_embedding",
        token=os.getenv("ASTRA_DB_TOKEN"),
        api_endpoint=os.getenv("DB_API_ENDPOINT"),
    )
    return vector_store


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )


try:
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to connect to Astra DB: {e}")
    st.stop()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided document context.
Use the context below to answer the user's question. If the answer is not in the context, say so politely.

Context:
{context}"""),
    ("human", "{question}")
])


chain = prompt_template | llm | StrOutputParser()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if question := st.chat_input("Ask a question about your PDF..."):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
           
            relevant_docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            response = chain.invoke({
                "context": context,
                "question": question
            })
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.markdown("## About")
    st.markdown("This chatbot answers questions based on a PDF document.")
    st.markdown("### How it works")
    st.markdown("1. A PDF was processed and stored in Astra DB")
    st.markdown("2. Each question finds relevant chunks from the document")
    st.markdown("3. Gemini generates answers based on those chunks")
    st.markdown("The document is about a platform named Digital AI Data Marketplace and Crowdsourcing, designed to aleviate the limited data availability for low resource languages in Ethiopia.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()