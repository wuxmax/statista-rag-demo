from typing import Optional

from loguru import logger

from statista_rag.models.data import TextEmbedding, RAGResponse
from statista_rag.rag.augmentation import create_context, create_references
from statista_rag.rag.generator import Generator
from statista_rag.rag.retriever import Retriever


class RAGPipeline:
    def __init__(self):
        self._generator = Generator()
        self._retriever = Retriever()

        self._DEFAULT_TEST_QUESTION: str = list(self._retriever.test_question_embeddings.keys())[0]

    def answer_question(self, question: Optional[str] = None) -> str:
        if question is None:
            question = self._DEFAULT_TEST_QUESTION
            logger.info(f"No question provided. Using default test question: '{question}'")

        retrieval_results: list[TextEmbedding] = self._retriever.retrieve_context(question)
        context: str = create_context(retrieval_results)
        references: str = create_references(retrieval_results)
        answer = self._generator.generate_answer(question, context=context)

        return RAGResponse(
            question=question,
            context=context,
            answer=answer,
            references=references
        )
