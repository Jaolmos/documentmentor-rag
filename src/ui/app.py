import streamlit as st
from pathlib import Path
from src.core.document_processor import DocumentProcessor
from src.data.vector_store import VectorStore
from src.core.qa_engine import QAEngine

class DocumentMentorUI:
    """Streamlit interface for DocumentMentor"""
    
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.qa_engine = QAEngine(self.vector_store)

    def display_chat(self):
        st.title("DocumentMentor")
        
        # Sidebar for document upload
        with st.sidebar:
            uploaded_file = st.file_uploader("", type=['pdf'])
            
            if uploaded_file:
                with st.spinner("Procesando..."):
                    temp_path = Path("data/processed") / uploaded_file.name
                    temp_path.write_bytes(uploaded_file.getvalue())
                    doc = self.processor.process_pdf(temp_path)
                    self.vector_store.add_document(doc)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if question := st.chat_input(""):
            # Add user message
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})

            # Get and display assistant response
            with st.chat_message("assistant"):
                response = self.qa_engine.get_answer(question)
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    app = DocumentMentorUI()
    app.display_chat()