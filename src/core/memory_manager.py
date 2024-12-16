from typing import Dict, List, Optional
from langchain_core.memory import BaseMemory
from langchain_community.memory import ConversationBufferMemory
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Conversation:
    """Represents a conversation with its metadata"""
    id: str
    title: str
    created_at: datetime
    context: str

class MemoryManager:
    """Manages conversation history and context"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.current_conversation: Optional[Conversation] = None
    
    def start_new_conversation(self, context: str, title: str = "Nueva conversación") -> None:
        self.memory.clear()
        self.current_conversation = Conversation(
            id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            title=title,
            created_at=datetime.now(),
            context=context
        )
    
    def add_interaction(self, user_input: str, assistant_response: str) -> None:
        if not self.current_conversation:
            raise ValueError("No hay una conversación activa")
            
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(assistant_response)
    
    def get_conversation_history(self) -> str:
        if not self.current_conversation:
            return ""
            
        messages = self.memory.chat_memory.messages
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    def clear_memory(self) -> None:
        self.memory.clear()
        self.current_conversation = None