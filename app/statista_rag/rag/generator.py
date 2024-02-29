from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from statista_rag.config import app_settings, rag_config, openai_api_config


class Generator:
    _llm_client: OpenAI
    _model: str

    _system_prompt: str = rag_config.generator_config.system_prompt
    _rag_prompt: str = rag_config.generator_config.rag_prompt

    def __init__(self, model: str = None):
        self.llm_client = OpenAI(
            base_url=openai_api_config.base_url,
            api_key=openai_api_config.key
        )

        self._model = model if model else openai_api_config.model

    def _query_llm(self, query: str, verbose: bool = False):
        if verbose:
            logger.info(f"Querying model '{self._model}' with:\n{query}")

        response: ChatCompletion = self.llm_client.chat.completions.create(
            model=self._model,
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self._system_prompt
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=query
                )
            ]
        )
        return response.choices[0].message.content

    def set_model(self, model: str):
        # This should actually query the available models and check if the model is available,
        # but the Ollama service does not support this API endpoint
        self._model = model

    def generate_answer(self, question: str, context: str, verbose: bool = False) -> str:
        ra_question: str = self._rag_prompt.format(question=question, context=context, verbose=verbose)
        return self._query_llm(ra_question, verbose=verbose)



