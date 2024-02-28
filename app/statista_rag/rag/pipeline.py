from typing import Optional

from loguru import logger

from statista_rag.models.data import TextEmbedding, RAGResponse
from statista_rag.rag.generator import Generator
from statista_rag.rag.retriever import Retriever
from statista_rag.rag.text_utils import create_context, create_references


class RAGPipeline:
    def __init__(self):
        self._generator = Generator()
        self._retriever = Retriever()

        self._DEFAULT_TEST_QUESTION_ID: int = 0

    def get_test_questions(self) -> dict[int, str]:
        return dict(enumerate(list(self._retriever.test_question_embeddings.keys())))

    def answer_question(self, question: Optional[str] = None, test_question_id: str = None) -> RAGResponse | None:
        if question is None and test_question_id is not None:
            question = list(self._retriever.test_question_embeddings.keys())[test_question_id]
            logger.info(f"Using test question {test_question_id}: '{question}'")

        if question is None and test_question_id is None:
            question = self._retriever.test_question_embeddings[test_question_id]
            logger.info(f"Using default test question '{self._DEFAULT_TEST_QUESTION_ID}': '{question}'")

        retrieval_results: list[TextEmbedding] = self._retriever.retrieve_context(question)
        if retrieval_results is None:
            return None

        context: str = create_context(retrieval_results)
        references: str = create_references(retrieval_results)
        answer = self._generator.generate_answer(question, context=context)

        return RAGResponse(
            question=question,
            context=context,
            answer=answer,
            references=references
        )
