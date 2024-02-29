from typing import Optional

from loguru import logger

from statista_rag.models.data import TextEmbedding, RAGResponse
from statista_rag.rag.generator import Generator
from statista_rag.rag.retriever import Retriever
from statista_rag.rag.augmentation import create_context, create_references


class RAGPipeline:
    def __init__(self):
        self._generator = Generator()
        self._retriever = Retriever()

        self._DEFAULT_TEST_QUESTION_ID: int = 0

    def _get_test_question(self, test_question_id: int) -> str:
        if test_question_id is not None:
            question = list(self._retriever.test_question_embeddings.keys())[test_question_id]
            logger.info(f"Using test question {test_question_id}: '{question}'")
            return question

        question = self._retriever.test_question_embeddings[self._DEFAULT_TEST_QUESTION_ID]
        logger.info(f"Using default test question '{self._DEFAULT_TEST_QUESTION_ID}': '{question}'")
        return question

    def get_test_questions(self) -> dict[int, str]:
        return dict(enumerate(list(self._retriever.test_question_embeddings.keys())))

    def set_rag_params(self, retriever_distance_measure: str = None, generator_model: int = None):
        if retriever_distance_measure:
            self._retriever.set_distance_measure(retriever_distance_measure)

        if generator_model:
            self._generator.set_model(generator_model)

    def answer_question(
            self, question: Optional[str] = None, test_question_id: str = None, verbose: bool = False
    ) -> RAGResponse | None:
        if not question:
            question = self._get_test_question(test_question_id)

        retrieval_results: list[TextEmbedding] = self._retriever.retrieve_context(question, verbose=verbose)
        if retrieval_results is None:
            return None

        context: str = create_context(retrieval_results)
        references: str = create_references(retrieval_results)

        answer: str = self._generator.generate_answer(question, context=context, verbose=verbose)

        return RAGResponse(
            question=question,
            answer=answer,
            references=references,
            context=context,
        )
