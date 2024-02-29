from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typer import Typer

from statista_rag.config import app_settings
from statista_rag.models.data import RAGResponse
from statista_rag.rag.pipeline import RAGPipeline

app = Typer()
console = Console()
rag_pipeline = RAGPipeline()


@app.command()
def answer(
        question: str = None,
        test_question_id: int = None,
        show_context: bool = False,
        generator_model: str = None,
        retriever_distance_measure: str = None,
        verbose: bool = False
):
    """
    Answer a question using the RAG model.

    :param question: The question to answer.
    :param test_question_id: The ID of a test question to use.
    :param show_context: Whether to show the context used to answer the question.
    :param generator_model: The model to use for generating the answer.
    :param retriever_distance_measure: The distance measure to use for retrieving the context.
    :param verbose: Whether to show verbose output.
    """
    rag_pipeline.set_rag_params(retriever_distance_measure, generator_model)

    response: RAGResponse = rag_pipeline.answer_question(question, test_question_id, verbose)
    if response is None:
        console.print("Could not answer the question.")
        return

    answer_parts: list = [
        ("Question: ", "bold"), response.question, "\n\n",
        ("Answer: ", "bold"), response.answer, "\n\n",
        ("References:", "bold"), "\n", response.references
    ]
    if show_context:
        answer_parts.extend(
            ["\n\n", ("Context:", "bold"), "\n", response.context]
        )

    console.print(Panel(Text.assemble(*answer_parts)))


@app.command()
def questions():
    """
    Show the available test questions.
    """
    test_questions = rag_pipeline.get_test_questions()
    formatted_answer = Text.assemble(
        ("Available test questions: ", "bold"), "\n",
        "\n".join(f"{qid}: {question}" for qid, question in test_questions.items())
    )
    console.print(Panel(formatted_answer))


