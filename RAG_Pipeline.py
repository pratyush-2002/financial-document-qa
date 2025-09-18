from langchain_community.document_loaders import PyMuPDFLoader,UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import os
import pandas as pd


llm=Ollama(model="mistral")

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def pdf_parser(file_path):
    pdf_loader=PyMuPDFLoader(file_path)
    pdf_doc=pdf_loader.load()
    return pdf_doc
def excel_parser(file_path):
    docs = []
    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        df= pd.read_excel (file_path, sheet_name=sheet_name)
        sheet_content = f"Sheet: {sheet_name}\n"
        sheet_content += f"Columns {', '.join(df.columns)}\n"
        sheet_content += f"Rows: {len(df)}\n\n"
        sheet_content += df.to_csv(index=False)

        doc= Document (
        page_content=sheet_content,
        metadata={
            'source': file_path,
            'sheet_name': sheet_name,
            'num_rows':len(df),
            'data_type':'excel sheet'
        })
        docs.append(doc)
    return docs
def text_cleaner(chunk):
    chunk=chunk.replace("\n"," ")
    chunk=" ".join(chunk.split())
    return chunk
def split(doc):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                            chunk_overlap=10)
    chunks=text_splitter.split_documents(doc)
    for c in chunks:
        c.page_content=text_cleaner(c.page_content)
    return chunks

def vector_db(chunks,embedding,collection):
    persist_directory="./chroma.db"
    cleaned_docs = filter_complex_metadata(chunks)
    vector_store=Chroma.from_documents(
        documents=cleaned_docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection
    )
    retriver=vector_store.as_retriever()
    return retriver
def file_processing(uploaded_file):
    save_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if uploaded_file.name.endswith(".pdf"):
        doc=pdf_parser(save_path)
    else:
        doc=excel_parser(save_path)
    chunks=split(doc)
    retriver=vector_db(chunks,embedding,collection=uploaded_file.name)
    return retriver

def create(retriver,question):
    system_prompt = """
    You are a highly skilled financial analyst AI assistant.

    Your role:
    - Answer questions strictly based on the provided financial documents (Excel or PDF).
    - Use ONLY the information in the retrieved context. Do not rely on outside knowledge.
    - If a required value (e.g., expense categories, revenue details) is missing, do NOT infer or approximate.

    Response rules:
    1. Be concise and clear, explaining in plain English.
    2. When presenting numbers, include units (e.g., USD) if available.
    3. Show calculations step by step if the user requests them or if the question involves growth rates, margins, or ratios.
    4. If the data required to compute an answer is incomplete or not present in the context, respond exactly with:
    "I couldn’t find that information in the provided documents."
    5. Never invent values, assume proxies, or fill in missing data.

    Context:
    {context}
    """

    prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human" ,"{input}")
    ]
    )
    stuff=create_stuff_documents_chain(llm,prompt)
    chain=create_retrieval_chain(retriver,stuff)
    response = chain.invoke({"input": question})
    print("Retrieved context:\n", response["context"])
    return response
