from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typer import Typer, Option

from statista_rag.models.data import RAGResponse
from statista_rag.rag.pipeline import RAGPipeline

app = Typer()
console = Console()
rag_pipeline = RAGPipeline()


@app.command()
def answer(
        question: str = Option(None, help="The question to answer."),
        test_question_id: int = Option(None, help="The ID of a test question to use."),
        show_context: bool = Option(False, help="Whether to show the context used to answer the question."),
        generator_model: str = Option(None, help="The model to use for generating the answer."),
        retriever_distance_measure: str = Option(
            None, help="The distance measure to use for retrieving the context."
        ),
        verbose: bool = Option(False, help="Whether to show verbose output.")
):
    """
    Answer a question using a Retrieval-Augmented-Generation chain.
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


