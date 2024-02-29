from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, YamlConfigSettingsSource


class AppSettings(BaseModel):
    app_dir: Path = Path(__file__).parents[1]
    pgvector_db_env_file: Path = app_dir.parent / 'pgvector_db.env'
    openai_api_env_file: Path = app_dir.parent / 'openai_api.env'

    data_dir: Path = app_dir.parent / 'data'
    question_embeddings_file_name: str = 'question_embedding.json'
    question_embeddings_file_path: Path = data_dir / question_embeddings_file_name

    rag_config_file_name: str = 'rag_config.yaml'
    rag_config_file_path: Path = app_dir / rag_config_file_name


app_settings = AppSettings()


class GeneratorConfig(BaseModel):
    system_prompt: str
    rag_prompt: str


class RetrieverConfig(BaseModel):
    embedding_table: str
    embedding_vector_length: int
    distance_measure: str
    top_k_results: int


class AugmentationConfig(BaseModel):
    context_separator: str
    use_only_last_context_part: bool
    statista_content_base_url: str


class RAGConfig(BaseSettings):
    generator_config: GeneratorConfig
    retriever_config: RetrieverConfig
    augmentation_config: AugmentationConfig
    model_config = SettingsConfigDict(yaml_file=app_settings.rag_config_file_path)

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


rag_config = RAGConfig()


class PGVectorDBConfig(BaseSettings):
    driver: str = 'postgresql+psycopg2'
    host: str
    port: int
    user: str
    password: str
    db_name: str = Field(alias='pgvector_db_name')

    @property
    def connection_string(self) -> str:
        return f'{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}'

    model_config = SettingsConfigDict(env_prefix='PGVECTOR_DB_', env_file=str(app_settings.pgvector_db_env_file))


pgvector_db_config = PGVectorDBConfig()


class OpenAIAPIConfig(BaseSettings):
    base_url: Optional[str] = None
    key: str
    model: str
    model_config = SettingsConfigDict(env_prefix='OPENAI_API_', env_file=str(app_settings.openai_api_env_file))


openai_api_config = OpenAIAPIConfig()


