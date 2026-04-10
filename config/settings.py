from pydantic_settings import  BaseSettings
from pathlib import Path

current_dir = Path(__file__).parent
file_path = f"{current_dir}/prompts/system_prompt.txt"
with open (file_path,'r',encoding='UTF-8') as f:
    content = f.read()

class Settings(BaseSettings):
    #LLM
    GROQ_API_KEY:str
    LLM_MODEL:str= 'openai/gpt-oss-120b'
    TEMPERATURE:float = 0.5
    MAX_TOKENs:int = 250

    SYSTEM_PROMPT:str =  content

    # Embedding Model
    EMBEDDING_MODEL:str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    CHUNK_SIZE:int = 700
    CHUNK_OVERLAP:int = 100

    # vectorstore
    VECTORSTORE_PATH:str = "data/vectorstore"

    # Retrieval
    TOP_K:int = 3

    # App
    APP_NAME:str = "RAG BOO"


    class Config:
        env_file = '.env'
        extra = "ignore"
    
settings = Settings()   