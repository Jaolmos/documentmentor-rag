import streamlit as st
from pathlib import Path
from src.core.document_processor import DocumentProcessor
from src.data.vector_store import VectorStore
from src.core.qa_engine import QAEngine

class DocumentMentorUI:
    """Streamlit interface for DocumentMentor"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.qa_engine = QAEngine(self.vector_store)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def display_chat(self):
        st.title("DocumentMentor")
        
        # Minimalist sidebar for document upload
        with st.sidebar:
            uploaded_file = st.file_uploader("", type=['pdf'])
            
            if uploaded_file:
                with st.spinner("Procesando..."):
                    temp_path = Path("data/processed") / uploaded_file.name
                    temp_path.write_bytes(uploaded_file.getvalue())
                    doc = self.processor.process_pdf(temp_path)
                    self.vector_store.add_document(doc)

        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(""):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                response = self.qa_engine.get_answer(prompt)
                st.markdown(response["answer"])
                
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )

if __name__ == "__main__":
    app = DocumentMentorUI()
    app.display_chat()