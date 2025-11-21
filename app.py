import os

import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_with_citations


# set the working directory
working_dir = os.path.dirname(os.path.abspath((__file__)))

st.title("gpt-3.5-turbo - Document RAG")

# Persist DB only in Streamlit session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# file uploader widget
uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    temp_paths = []
    for uf in uploaded_files:
        path = f"tmp_{uf.name}"
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        temp_paths.append(path)

    st.session_state.vectordb = process_document_to_chroma_db(temp_paths)
    st.success(f"Processed {len(temp_paths)} documents into vector DB.")

# text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    if st.session_state.vectordb is None:
        st.warning("Upload PDFs first.")
    elif not user_question.strip():
        st.warning("Enter a question.")
    else:
        answer, sources = answer_with_citations(user_question, st.session_state.vectordb)
        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        for s in sources:
            st.write(f"- **{s['source']}** (page {s['page']})")
