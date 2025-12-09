import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuTalk AI", page_icon="🤖")

# --- 2. SIDEBAR (API KEY INPUT) ---
with st.sidebar:
    st.title("🤖 DocuTalk Settings")
    st.markdown("This smart tool allows you to chat with any PDF document.")
    
    # Securely getting the API key from the user
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF.")
    st.markdown("2. System converts text to numbers (Embeddings).")
    st.markdown("3. AI finds the relevant section and answers you.")
    
    if not api_key:
        st.warning("⚠️ Please enter an API Key to proceed.")

# --- 3. HELPER FUNCTIONS ---

def get_pdf_text(pdf_docs):
    """
    Extracts text from uploaded PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the massive text into smaller, manageable chunks for the AI.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, # Overlap helps keep context between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """
    Converts text chunks into vectors (numbers) and stores them in FAISS.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # Save locally for speed (optional, but good practice)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(api_key):
    """
    Sets up the AI model (LLM) to answer questions based on context.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def user_input(user_question, api_key):
    """
    Handles the user's question:
    1. Loads the vector database.
    2. Searches for similar content.
    3. Sends content + question to GPT.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Load the vector DB (Allow dangerous deserialization as we created the file)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Search the DB for text relevant to the question
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain(api_key)
    
    with st.spinner("Thinking..."):
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("### 🤖 Answer:")
        st.write(response["output_text"])

# --- 4. MAIN USER INTERFACE ---

def main():
    st.header("Chat with your PDF 📚")
    
    # Step 1: User asks a question
    user_question = st.text_input("Ask a question about your document:")

    if user_question and api_key:
        user_input(user_question, api_key)

    # Step 2: User uploads files
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process Docs"):
            if not api_key:
                st.error("Please insert API Key first.")
            else:
                with st.spinner("Processing... (This puts the 'Smart' in Smart Project)"):
                    # A. Extract Text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # B. Split Text
                    text_chunks = get_text_chunks(raw_text)
                    
                    # C. Create Vectors
                    get_vector_store(text_chunks, api_key)
                    
                    st.success("Done! You can now ask questions.")

if __name__ == "__main__":
    main()