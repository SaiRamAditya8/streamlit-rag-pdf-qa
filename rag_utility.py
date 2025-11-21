import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
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


def process_document_to_chroma_db(pdf_paths, chunk_size=1500, chunk_overlap=200):
    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(pdf_path)
            d.metadata["page"] = d.metadata.get("page", "unknown")

        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)

    vectordb = Chroma.from_documents(all_chunks, embedding)
    return vectordb


def answer_with_citations(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": question})

    answer = result["result"]
    docs = result["source_documents"]

    sources = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "unknown")
        })

    return answer, sources
