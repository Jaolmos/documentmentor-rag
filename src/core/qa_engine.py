from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
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
        self.memory = ConversationBufferMemory()
        
    def get_answer(self, question: str) -> dict:
        relevant_chunks = self.vector_store.search(question)
        context = "\n".join([chunk["chunk"] for chunk in relevant_chunks])
        
        conversation_history = self.memory.load_memory_variables({})
        
        prompt = f"""
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
        
        Conversation History:
        {conversation_history}
        
        Question: {question}
        """
        
        response = self.llm.predict(prompt)
        
        self.memory.save_context(
            {"input": question},
            {"output": response}
        )
        
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