from statista_rag.models.data import TextEmbedding


_CONTEXT_SEPARATOR: str = '#'
_STATISTA_CONTENT_BASE_URL: str = "https://www.statista.com/statistics"


def create_references(text_embeddings: list[TextEmbedding]) -> str:
    references: list[str] = []
    for i, text_embedding in enumerate(text_embeddings):
        text_parts: list[str] = text_embedding.text.split(_CONTEXT_SEPARATOR)
        reference_name: str = text_parts[2] if len(text_parts) > 2 else text_parts[0]
        reference_link: str = f"{_STATISTA_CONTENT_BASE_URL}/{text_embedding.content_id}"
        references.append(f"[{i + 1}] {reference_name} ({reference_link})")
    return "\n".join(references)


def create_context(text_embeddings: list[TextEmbedding]) -> str:
    context_pieces: list[str] = [
        f"{i + 1}) {text_embedding.text}" for i, text_embedding in enumerate(text_embeddings)
    ]
    return "\n".join(context_pieces)
