import logging
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils.config import OPENAI_API_KEY
from src.data.vector_store import VectorStore

logger = logging.getLogger(__name__)

class QAEngine:
    """Handles document-based question answering with conversation memory"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY,
            request_timeout=60
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
        Eres un asistente técnico especializado en tecnología y desarrollo de software que SIEMPRE responde en español. 
        Tu experiencia abarca:
        - Lenguajes de programación y desarrollo de software
        - Conceptos fundamentales de programación
        - Estructuras de datos y algoritmos
        - Bases de datos y gestión de datos
        - Frameworks y bibliotecas populares
        - Arquitectura de software
        - Desarrollo web (frontend y backend)
        - Buenas prácticas de programación

        Tu personalidad es profesional pero cercana, y te caracterizas por:
        - Explicar conceptos técnicos de manera clara y comprensible
        - Proporcionar ejemplos prácticos cuando sea posible
        - Usar analogías para facilitar la comprensión
        - Mantener un equilibrio entre tecnicismo y claridad
        - Ser paciente y detallado en las explicaciones

        Directrices:
        - SIEMPRE responde en español
        - Si se menciona código en el contexto, inclúyelo en la explicación
        - Explica los conceptos técnicos de forma clara pero precisa
        - Si algo no está en el contexto, dilo honestamente
        - Usa terminología técnica apropiadamente
        - Si es relevante, menciona qué parte de la documentación contiene la información
        - Sé amigable y cercano, pero mantén la profesionalidad

        Contexto:
        {context}
        
        Pregunta: {question}
        """)
        
        self.qa_chain = (
            {
                "context": lambda x: self._get_context(x["question"]),
                "question": RunnablePassthrough()
            }
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )
        
    def _get_context(self, question: str) -> str:
        """Get context from vector store with error handling"""
        try:
            results = self.vector_store.search(question)
            return "\n".join([chunk["chunk"] for chunk in results])
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""

    def get_answer(self, question: str) -> Dict[str, str]:
        """Get answer with error handling and logging"""
        try:
            logger.info(f"Processing question: {question}")
            answer = self.qa_chain.invoke({"question": question})
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Error getting answer: {e}")
            return {
                "error": "Lo siento, hubo un error procesando tu pregunta. Por favor, intenta de nuevo."
            }

    def get_initial_message(self) -> str:
        """Returns the initial greeting message"""
        return """¡Hola! 👋 Soy tu asistente técnico personal. 
        
Estoy aquí para ayudarte a entender documentación sobre desarrollo de software, inteligencia artificial, bases de datos, frameworks, arquitectura de sistemas y tecnologías relacionadas.


Características:
- Memoria persistente: Consulto todos los documentos subidos anteriormente
- Análisis contextual: Respuestas basadas en tu documentación
- Ejemplos prácticos cuando sea posible

Para empezar:
1. Sube un documento PDF técnico en el panel lateral
2. Hazme preguntas sobre cualquier documento
3. ¡Te ayudaré a entenderlo!

¿En qué puedo ayudarte hoy?"""