from loguru import logger
from sqlalchemy import Engine
from sqlmodel import Session, create_engine, select
from sqlmodel.sql.expression import Select

from statista_rag.config import rag_config, pgvector_db_config
from statista_rag.models.data import TextEmbeddingMap, TextEmbedding
from statista_rag.utils import load_text_embeddings


class Retriever:
    _db_engine: Engine

    _AVAILABLE_DISTANCE_MEASURES: list[str] = [
        "l2_distance",
        "cosine_distance",
        "max_inner_product"
    ]

    _top_k_results: int = rag_config.retriever_config.top_k_results
    _distance_measure_name: str = rag_config.retriever_config.distance_measure

    test_question_embeddings: TextEmbeddingMap

    def __init__(self):
        self._db_engine = create_engine(pgvector_db_config.connection_string)
        self.test_question_embeddings = load_text_embeddings()
        self.set_distance_measure(self._distance_measure_name)

    def set_distance_measure(self, distance_measure_name: str):
        assert distance_measure_name in self._AVAILABLE_DISTANCE_MEASURES, (
            f"Distance measure {distance_measure_name} not supported."
            f"Supported distance measures: {', '.join(self._AVAILABLE_DISTANCE_MEASURES)}."
        )
        self._distance_measure_name = distance_measure_name

    def retrieve_context(self, question: str, verbose: bool = False) -> list[TextEmbedding] | None:
        if question not in self.test_question_embeddings:
            logger.info(
                f"Currently only the following test questions are supported:\n"
                + "\n".join(f"- {test_question}" for test_question in self.test_question_embeddings.keys())
            )
            return None

        return self.retrieve_similar_text_embeddings(self.test_question_embeddings[question], verbose=verbose)

    def retrieve_similar_text_embeddings(self, embedding: list[float], verbose: bool = False) -> list[TextEmbedding]:
        distance_measure = getattr(TextEmbedding.embedding_vector, self._distance_measure_name)

        with Session(self._db_engine) as session:
            query: Select = select(
                TextEmbedding, distance_measure(embedding).label('distance')
            ).order_by(distance_measure(embedding)).limit(self._top_k_results)

            if verbose:
                logger.info(f"Executing vector search query...")
            results = session.exec(query).all()

            similarity_search_results = [(result.TextEmbedding, result.distance) for result in results]
            if verbose:
                logger.info(
                    f"Vector search results by '{self._distance_measure_name}':\n"
                    + "\n---\n".join(
                        f"- {result[0].text}\n-> distance {result[1]}" for result in similarity_search_results
                    )
                )

        return [similarity_search_result[0] for similarity_search_result in similarity_search_results]

