import os

import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question


# set the working directory
working_dir = os.path.dirname(os.path.abspath((__file__)))

st.title("gpt-3.5-turbo - Document RAG")

# Persist DB only in Streamlit session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    #file path to save the uploaded file
    tmp_path = os.path.join(working_dir, uploaded_file.name)
    #  save the file
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.vectordb = process_document_to_chroma_db(tmp_path)
    st.info("Document Processed Successfully")

# text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):

    answer = answer_question(user_question,st.session_state.vectordb)

    st.markdown("### gpt-3.5-turbo Response")
    st.markdown(answer)
