from src.chunker.chunker import make_chunks
from src.pipelines.embedding_pipeline import build_vectorstore
from dotenv import load_dotenv
load_dotenv()

chunks = make_chunks()
vector_store = build_vectorstore(chunks)