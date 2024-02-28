from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from statista_rag.config import OpenAIAPIConfig


class Generator:
    _openai_api_config: OpenAIAPIConfig
    _llm_client: OpenAI

    _SYSTEM_PROMPT: str = """
    You are an assistant for question-answering tasks. 
    """

    _RAG_PROMPT: str = """
    Answer the following question. 
    Use only the pieces of retrieved context to answer the question.
    You do not have to use all of the context pieces to answer the question.
    Be as specific and as concise as possible while answering the question.
    The context pieces are sorted by relevance, with the most relevant piece first.
    Add a reference to a context piece to the parts of your answer, that are based on it.
    The references should be added in square brackets after a sentence, like so: [0]
    Do not repeat the pieces of context in your answer. Do not create a reference section.
    If you cannot answer the question based on the provided context, then say so.

    Question: {question}
    Context Pieces: {context}
    """

    def __init__(self):
        self._openai_api_config = OpenAIAPIConfig()

        self.llm_client = OpenAI(
            base_url=self._openai_api_config.base_url,
            api_key=self._openai_api_config.key
        )

    def query_llm(self, query: str):
        response: ChatCompletion = self.llm_client.chat.completions.create(
            model=self._openai_api_config.model,
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self._SYSTEM_PROMPT
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=query
                )
            ]
        )
        return response.choices[0].message.content

    def generate_answer(self, question: str, context: str) -> str:
        ra_question: str = self._RAG_PROMPT.format(question=question, context=context)
        return self.query_llm(ra_question)



