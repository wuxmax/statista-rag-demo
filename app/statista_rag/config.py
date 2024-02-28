from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings:
    app_dir: Path = Path(__file__).parents[1]
    pgvector_db_env_file: Path = app_dir.parent / 'pgvector_db.env'
    openai_api_env_file: Path = app_dir.parent / 'openai_api.env'

    data_dir: Path = app_dir.parent / 'data'
    question_embeddings_file: Path = data_dir / 'question_embedding.json'


class PGVectorTables:
    embedding_table: str = 'embeddings'
    statistics_table: str = 'statistics'
    statistics_embeddings_mapping_table: str = 'statistics_embeddings'


class PGVectorDBConnection(BaseSettings):
    driver: str = 'postgresql+psycopg2'
    host: str
    port: int
    user: str
    password: str
    db_name: str = Field(alias='pgvector_db_name')

    @property
    def connection_string(self) -> str:
        return f'{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}'

    model_config = SettingsConfigDict(env_prefix='PGVECTOR_DB_', env_file=str(AppSettings.pgvector_db_env_file))


class OpenAIAPIConfig(BaseSettings):
    base_url: Optional[str] = None
    key: str
    model: str
    model_config = SettingsConfigDict(env_prefix='OPENAI_API_', env_file=str(AppSettings.openai_api_env_file))
