from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils.config import OPENAI_API_KEY
from src.data.vector_store import VectorStore

class QAEngine:
    """Handles document-based question answering with conversation memory"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Define el prompt template usando el nuevo formato
        self.prompt = ChatPromptTemplate.from_template("""
        You are a technical documentation assistant specialized in programming, software development, and computer science.
        Your task is to provide clear, accurate, and technical answers based on the provided documentation.

        Guidelines:
        - If code is mentioned in the context, include it in the explanation
        - Explain technical concepts clearly but maintain technical accuracy
        - If something is not in the context, say so and don't make assumptions
        - Use technical terminology appropriately
        - If relevant, mention which part of the documentation the information comes from

        Context:
        {context}
        
        Question: {question}
        """)
        
        # Configura la cadena RAG usando LCEL (LangChain Expression Language)
        self.qa_chain = (
            {
                "context": lambda x: "\n".join([chunk["chunk"] for chunk in self.vector_store.search(x["question"])]),
                "question": RunnablePassthrough()
            }
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )
        
    def get_answer(self, question: str) -> dict:
        """
        Get answer for a question using RAG (Retrieval Augmented Generation)
        """
        # Obtiene los chunks relevantes
        relevant_chunks = self.vector_store.search(question)
        
        # Genera la respuesta usando la cadena RAG
        response = self.qa_chain.invoke({"question": question})
        
        return {
            "answer": response,
            "sources": [
                {
                    "title": chunk["title"],
                    "content": chunk["chunk"][:200] + "..."
                }
                for chunk in relevant_chunks
            ],
            "confidence": relevant_chunks[0]["score"] if relevant_chunks else 0
        }