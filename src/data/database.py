from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from typing import List, Optional
from src.utils.config import DATABASE_URL

Base = declarative_base()

class Document(Base):
    """Database model for stored documents"""
    __tablename__ = 'documents'

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String, nullable=False)

class Database:
    """Handles database operations for document storage"""
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_document(self, doc_id: str, title: str, content: str, file_path: str) -> None:
        session = self.Session()
        try:
            document = Document(
                id=doc_id,
                title=title,
                content=content,
                file_path=file_path
            )
            session.add(document)
            session.commit()
        finally:
            session.close()
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        session = self.Session()
        try:
            return session.query(Document).filter(Document.id == doc_id).first()
        finally:
            session.close()
    
    def get_all_documents(self) -> List[Document]:
        session = self.Session()
        try:
            return session.query(Document).all()
        finally:
            session.close()
    
    def delete_document(self, doc_id: str) -> bool:
        session = self.Session()
        try:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if document:
                session.delete(document)
                session.commit()
                return True
            return False
        finally:
            session.close()