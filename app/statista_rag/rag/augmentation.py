from statista_rag.config import rag_config
from statista_rag.models.data import TextEmbedding


def create_references(text_embeddings: list[TextEmbedding]) -> str:
    references: list[str] = []
    for i, text_embedding in enumerate(text_embeddings):
        text_parts: list[str] = text_embedding.text.split(rag_config.augmentation_config.context_separator)
        reference_name: str = text_parts[2] if len(text_parts) > 2 else text_parts[0]
        reference_link: str = f"{rag_config.augmentation_config.statista_content_base_url}/{text_embedding.content_id}"
        references.append(f"[{i + 1}] {reference_name} ({reference_link})")
    return "\n".join(references)


def create_context(text_embeddings: list[TextEmbedding]) -> str:
    context_piece_texts: list[str] = [
        text_embedding.text.split(rag_config.augmentation_config.context_separator)[-1]
        if rag_config.augmentation_config.use_only_last_context_part
        else text_embedding.text
        for text_embedding in text_embeddings
    ]
    context_pieces: list[str] = [
        f"{i + 1}) {context_piece_text}" for i, context_piece_text
        in enumerate(context_piece_texts)
    ]
    return "\n".join(context_pieces)
