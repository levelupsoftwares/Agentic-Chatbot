from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import settings



def build_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(
        model_name = settings.EMBEDDING_MODEL,    
    )

    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding=embedding,
        persist_directory= 'data/vectorstore',
        collection_name='first_embedding'
    )

    return vectorstore

