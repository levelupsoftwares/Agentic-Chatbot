# from langchain_community.retrievers import BM25Retriever
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from dotenv import load_dotenv
load_dotenv()


def load_vectorstore():
    embedding =HuggingFaceEmbeddings(model=settings.EMBEDDING_MODEL) 

    vectorstore = Chroma(
        persist_directory = 'data/vectorstore',
        collection_name = 'first_embedding',
        embedding_function= embedding
    )
    return vectorstore

def get_retriever():
    vector_store = load_vectorstore()
    return vector_store.as_retriever(
        search_kwargs={"k":settings.TOP_K}
    )

if __name__ == "__main__":
    retriever = get_retriever()
    

    