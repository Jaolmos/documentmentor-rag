from typing import Dict, List, Optional
from langchain.memory import ConversationBufferMemory
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
        self.memory = ConversationBufferMemory()
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
            
        self.memory.save_context(
            {"input": user_input},
            {"output": assistant_response}
        )
    
    def get_conversation_history(self) -> str:
        if not self.current_conversation:
            return ""
            
        return self.memory.load_memory_variables({}).get("history", "")
    
    def clear_memory(self) -> None:
        self.memory.clear()
        self.current_conversation = None