from loguru import logger
from sqlalchemy import Engine
from sqlmodel import Session, create_engine, select
from sqlmodel.sql.expression import Select

from statista_rag.config import PGVectorDBConnection
from statista_rag.models.data import TextEmbeddingMap, TextEmbedding
from statista_rag.utils import load_text_embeddings


class Retriever:
    db_connection: PGVectorDBConnection
    db_engine: Engine

    test_question_embeddings: TextEmbeddingMap

    _TOP_K_RESULTS: int = 5
    _DISTANCE_MEASURES: list[str] = [
        "l2_distance",
        "cosine_distance",
        "max_inner_product"
    ]

    def __init__(self):
        self.db_connection = PGVectorDBConnection()
        self.db_engine = create_engine(self.db_connection.connection_string)
        self.test_question_embeddings = load_text_embeddings()

    def retrieve_context(self, question: str) -> list[TextEmbedding] | None:
        if question not in self.test_question_embeddings:
            logger.info(
                f"Currently only the following test questions are supported:\n"
                + "\n".join(f"- {test_question}" for test_question in self.test_question_embeddings.keys())
            )
            return None

        return self.retrieve_by_embedding(self.test_question_embeddings[question])

    def retrieve_by_embedding(
            self,
            embedding: list[float],
            distance_measure_name: str = _DISTANCE_MEASURES[1]
    ) -> list[TextEmbedding]:
        assert distance_measure_name in self._DISTANCE_MEASURES, (
            f"Distance measure {distance_measure_name} not supported"
        )

        distance_measure = getattr(TextEmbedding.embedding_vector, distance_measure_name)

        with Session(self.db_engine) as session:
            statement: Select = select(TextEmbedding).order_by(distance_measure(embedding)).limit(self._TOP_K_RESULTS)
            similarity_search_results: list[TextEmbedding] = session.exec(statement).all()

        return similarity_search_results

