import streamlit as st
from RAG_Pipeline import file_processing,create

st.title("Financial Q&A")
uploaded_file = st.file_uploader("Upload your Excel or PDF", type=["xlsx", "pdf"])
if uploaded_file is not None:
    st.info("Processing your file...")
    retriever = file_processing(uploaded_file)
    st.session_state["retriever"] = retriever
    st.success("File processed successfully.")

question = st.text_input("Ask a question about the uploaded document:")

if question:
    if "retriever" in st.session_state:
        answer = create(st.session_state["retriever"], question)
        # If the chain returns a dictionary with 'answer'
        if isinstance(answer, dict) and "answer" in answer:
            st.write("### Answer")
            st.write(answer["answer"])
        else:
            # fallback if chain returns just text
            st.write("### Answer")
            st.write(answer)
    else:
        st.warning("Please upload a document first!")