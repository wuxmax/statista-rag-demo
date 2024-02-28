from statista_rag.models.data import TextEmbedding


def create_references(text_embeddings: list[TextEmbedding]) -> str:
    references: list[str] = [
        f"[{i + 1}] {text_embedding.text}" for i, text_embedding in enumerate(text_embeddings)
    ]
    return "\n".join(references)


def create_context(text_embeddings: list[TextEmbedding]) -> str:
    context_pieces: list[str] = [
        f"{i + 1}) {text_embedding.text}" for i, text_embedding in enumerate(text_embeddings)
    ]
    return "\n".join(context_pieces)
