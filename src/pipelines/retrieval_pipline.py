from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from src.chunker.chunker import make_chunks
from config.settings import settings
from dotenv import load_dotenv
load_dotenv()

chunks =make_chunks()

def load_vectorstore():
    embedding =HuggingFaceEmbeddings(model=settings.EMBEDDING_MODEL) 

    vectorstore = Chroma(
        persist_directory = 'data/vectorstore',
        collection_name = 'first_embedding',
        embedding_function= embedding
    )
    return vectorstore

def get_retriever(chunks):
    vector_store = load_vectorstore()

             # Dense search    
    dense_search =  vector_store.as_retriever(
        search_kwargs={"k":settings.TOP_K ,"fetch_k":10 },
        search_type='mmr'
    )

            # Sparse search
    sparse_search = BM25Retriever.from_documents(
                                documents=chunks,
                                kwargs={"k":settings.TOP_K}
                                                 )

    return  dense_search, sparse_search 


def hybrid_search():
    dense_search ,sparse_search = get_retriever(chunks)
    return EnsembleRetriever(
        retrievers=[dense_search ,sparse_search],
        weights=[0.5,0.5]
    )

if __name__ == "__main__":
    retriever = hybrid_search()
    # resutl  = retriever.invoke("what services are you offering")
    # print(resutl)

    