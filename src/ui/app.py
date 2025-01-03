import streamlit as st
from pathlib import Path
from src.core.document_processor import DocumentProcessor
from src.data.vector_store import VectorStore
from src.core.qa_engine import QAEngine
from src.data.database import Database

class DocumentMentorUI:
    """Streamlit interface for DocumentMentor"""
    
    def __init__(self):
        self.database = Database()
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.qa_engine = QAEngine(self.vector_store)
        
        # Inicializar el estado de los mensajes con el saludo inicial
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": self.qa_engine.get_initial_message()
            }]

    def display_chat(self):
        """Main chat display interface"""
        st.title("DocumentMentor")
        
        with st.sidebar:
            # Initialize upload state
            if "upload_state" not in st.session_state:
                st.session_state.upload_state = False
            
            def handle_upload():
                if st.session_state.uploaded_file is not None:
                    st.session_state.upload_state = True
            
            uploaded_file = st.file_uploader(
                "Cargar documento PDF",
                type=['pdf'],
                label_visibility="collapsed",
                key="uploaded_file",
                on_change=handle_upload
            )
            
            if uploaded_file and st.session_state.upload_state:
                with st.spinner("Procesando..."):
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
                        st.session_state.upload_state = False
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.upload_state = False
        
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