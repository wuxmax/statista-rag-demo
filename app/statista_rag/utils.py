from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from statista_rag.config import app_settings
from statista_rag.models.data import TextEmbeddingMap


text_embedding_ta = TypeAdapter(TextEmbeddingMap)


def load_text_embeddings(
        embeddings_file: Path = app_settings.question_embeddings_file_path
) -> TextEmbeddingMap | None:
    try:
        return text_embedding_ta.validate_json(embeddings_file.read_text())
    except Exception as e:
        logger.error(f'Error loading embeddings from {embeddings_file}: {e}')
        return None
