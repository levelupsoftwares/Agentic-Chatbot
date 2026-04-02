from pydantic_settings import  BaseSettings

class Settings(BaseSettings):
    #LLM
    GROQ_API_KEY:str
    LLM_MODEL:str= 'openai/gpt-oss-120b'
    TEMPERATURE:float = 0.5
    MAX_TOKENs:int = 250

    # Embedding Model
    EMBEDDING_MODEL:str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    CHUNK_SIZE:int = 500
    CHUNK_OVERLAP:int = 100

    # vectorstore
    VECTORSTORE_PATH:str = "data/vectorstore"

    # Retrieval
    TOP_K:int = 4

    # App
    APP_NAME:str = "RAG BOO"


    class Config:
        env_file = '.env'
        extra = "ignore"
    
settings = Settings()   