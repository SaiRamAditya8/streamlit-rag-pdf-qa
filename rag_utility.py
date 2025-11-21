import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


# Load environment variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

# Load the embedding model
embedding = HuggingFaceEmbeddings()

# Load the gpt-3.5-turbo model from OpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)


def process_document_to_chroma_db(file_path):
    # Load the PDF document using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    # Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    # Store the document chunks in a Chroma vector database
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding
    )
    return vectordb


def answer_question(user_question,vectordb):
    # Create a retriever for document search
    retriever = vectordb.as_retriever()

    # Create a RetrievalQA chain to answer user questions using Llama-3.3-70B
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer
