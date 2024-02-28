from rich.console import Console
from rich.markdown import Markdown

from statista_rag.models.data import RAGResponse
from statista_rag.rag.pipeline import RAGPipeline


def main(question: str = None):
    """
    Answer a question using the RAG model.

    :param question: The question to answer.
    """

    console = Console()
    rag_pipeline = RAGPipeline()

    response: RAGResponse = rag_pipeline.answer_question(question)

    response_md = Markdown(
        f"**Question:** {question}\n\n"
        f"**Answer:** {response.answer}\n\n"
        f"**References:** {response.references}"
    )
    console.print(response_md)




