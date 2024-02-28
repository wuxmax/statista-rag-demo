from typing import Optional

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlmodel import SQLModel, Field
from sqlalchemy import Column


EMBEDDING_VECTOR_SIZE: int = 1536


TextEmbeddingMap = dict[str, list[float]]


class TextEmbedding(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    __tablename__ = 'embeddings'

    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    embedding_vector: list[float] = Field(sa_column=Column(Vector(EMBEDDING_VECTOR_SIZE)))
    content_type: str
    content_id: int


class RAGResponse(BaseModel):
    question: str
    context: str
    answer: str
    references: str

