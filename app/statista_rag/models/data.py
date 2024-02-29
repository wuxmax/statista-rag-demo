from collections import OrderedDict
from typing import Optional

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlmodel import SQLModel, Field
from sqlalchemy import Column

from statista_rag.config import rag_config

TextEmbeddingMap = OrderedDict[str, list[float]]


class TextEmbedding(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    __tablename__ = rag_config.retriever_config.embedding_table

    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    embedding_vector: list[float] = Field(
        sa_column=Column(Vector(rag_config.retriever_config.embedding_vector_length))
    )
    content_type: str
    content_id: int


class RAGResponse(BaseModel):
    question: str
    context: str
    answer: str
    references: str


