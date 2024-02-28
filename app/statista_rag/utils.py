import logging

from pathlib import Path

from pydantic import TypeAdapter

from statista_rag.config import AppSettings
from statista_rag.models.data import TextEmbeddingMap

logger: logging.Logger = logging.getLogger()


# TextEmbeddings = RootModel[list[TextEmbedding]]

text_embedding_ta = TypeAdapter(TextEmbeddingMap)


def load_text_embeddings(embeddings_file: Path = AppSettings.question_embeddings_file) -> TextEmbeddingMap | None:
    try:
        return text_embedding_ta.validate_json(embeddings_file.read_text())
    except Exception as e:
        logger.error(f'Error loading embeddings from {embeddings_file}: {e}')
        return None
