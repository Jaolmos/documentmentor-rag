import streamlit as st
from pathlib import Path
from src.core.document_processor import DocumentProcessor
from src.data.vector_store import VectorStore
from src.core.qa_engine import QAEngine
from src.data.database import Database

class DocumentMentorUI:
    """Streamlit interface for DocumentMentor"""
    
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        self.database = Database()
        
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.qa_engine = QAEngine(self.vector_store)

    def display_chat(self):
        st.title("DocumentMentor")
        
        with st.sidebar:
            uploaded_file = st.file_uploader(
                "Cargar documento PDF",
                type=['pdf'],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                with st.spinner("Procesando documento..."):
                    try:
                        temp_path = Path("data/processed") / uploaded_file.name
                        temp_path.write_bytes(uploaded_file.getvalue())
                        doc = self.processor.process_pdf(temp_path)
                        
                        self.database.save_document(
                            doc_id=doc.id,
                            title=doc.title,
                            content=doc.content,
                            file_path=str(doc.source_path)
                        )
                        
                        self.vector_store.add_document(doc)
                        st.success("Documento procesado correctamente")
                    except Exception as e:
                        st.error(f"Error procesando documento: {e}")
        
        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question := st.chat_input("Escribe tu pregunta aquí"):
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                with st.spinner("Procesando respuesta..."):
                    try:
                        response = self.qa_engine.get_answer(question)
                        if "error" in response:
                            st.error(response["error"])
                        else:
                            st.markdown(response["answer"])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response["answer"]
                            })
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    app = DocumentMentorUI()
    app.display_chat()