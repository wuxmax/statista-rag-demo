# Statista RAG Demo

This is a demo CLI app which implements a simple Retrieval-Augmented-Generation (RAG) pipeline. 
The app takes a question as input and queries an LLM to generate the answer based on the retrieved context.

The app retrieves the most relevant content pieces for a given question based on embedding similarity.
It then uses the question and the retrieved context in a specific prompt to an LLM to generate the answer. 
It currently only works with a set of pre-defined questions and their corresponding embeddings, 
due to a lack of access to an embedding model.

## Setup

### External Services
- **PGVector Database**: The app requires a PostgreSQL database with the pgvector extension installed 
and populated with the necessary data. The contained `docker-compose.yml` file can be used to set up a local database.
- **OpenAI API compatible LLM endpoint**: The app requires an endpoint to a language model
that implements the OpenAI API chat completion endpoint: https://platform.openai.com/docs/api-reference/chat/create
  (The ollama local LLM service was used for testing: https://github.com/ollama/ollama)

### Local Setup

#### Requirements
Python 3.12.2+ (Project is build on 3.12.2, but should work with 3.10+)

#### Installation
1. Clone the repository
2. Change into the project directory
    ````bash
    cd statista-rag-demo
    ````
3. Install the required packages (preferably in a virtual environment)
    ```bash
    pip install -r requirements.txt
    ```

### Configure external services
The external services have to be configured via environment variables. They can either be set in the environment 
or in a corresponding `<service>.env` file in the root directory.

#### Database 
Example `pgvector_db.env` file:
```env
PGVECTOR_DB_NAME=statista-rag-demo
PGVECTOR_DB_USER=USER123
PGVECTOR_DB_PASSWORD=PW12345
PGVECTOR_DB_HOST=localhost
PGVECTOR_DB_PORT=5432
```

#### OpenAI API
Example `openai_api.env` file:
```env
OPENAI_API_BASE_URL="localhost:8080/v1"
OPENAI_API_KEY="xxx"
OPENAI_API_MODEL="mistral:7b-instruct-v0.2-q8_0"
```
The `OPENAI_API_BASE_URL` parameter is optional and default to the actual OpenAI API endpoint.  
The `0PENAI_API_MODEL` parameter must be set to a model that is available at the specified endpoint.

## Run the app
The app can be run with the following command from the project root directory:
```bash
python app/main.py answer --question "What is the revenue of Alphabet Inc?"
```
The additional options for the `answer` command can be displayed with the `--help` option:
```bash
python app/main.py answer --help 
                                                                                                                                                                       
 Usage: main.py answer [OPTIONS]                                                                                                                                       
                                                                                                                                                                       
 Answer a question using a Retrieval-Augmented-Generation chain.                                                                                                       
                                                                                                                                                                       
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --question                                           TEXT     The question to answer. [default: None]                                             │
│ --test-question-id                                   INTEGER  The ID of a test question to use. [default: None]                                   │
│ --show-context                  --no-show-context             Whether to show the context used to answer the question. [default: no-show-context] │
│ --generator-model                                    TEXT     The model to use for generating the answer. [default: None]                         │
│ --retriever-distance-measure                         TEXT     The distance measure to use for retrieving the context. [default: None]             │
│ --verbose                       --no-verbose                  Whether to show verbose output. [default: no-verbose]                               │
│ --help                                                        Show this message and exit.                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
If no question and no test question ID is provided, the first test question will be used.


### Show the available test questions
The available test questions can be displayed with the following command:
```bash
python app/main.py questions
```

### Configure the RAG pipeline
The RAG pipeline is configured via the `app/rag_config.yaml` file. The following parameters can be set:

- `generator_config`:
    - `system_promp`: Sets the assistant's role for answering questions.
    - `rag_prompt`: Provides a detailed template for generating answers using retrieved context.
- `retriever_config`:
    - `embedding_table`: Specifies the database table for embeddings.
    - `embedding_vector_length`: Defines the length of embedding vectors.
    - `distance_measure`: Determines the method for calculating similarity between question and context embeddings.
    - `top_k_results`: Number of top results to retrieve for answering.
- `augmentation_config`:
  - `context_separator`: Character used to separate context pieces.
  - `use_only_last_context_part`: Boolean to indicate if only the last part of the context should be used.
  - `statista_content_base_url`: Base URL for displaying links to the referenced Statista content.