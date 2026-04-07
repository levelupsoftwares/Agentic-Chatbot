from langchain_text_splitters import  RecursiveCharacterTextSplitter
from src.loaders.textLoader import load_documents
from config.settings import settings

# splitter
def chunk_documents(documents):
   splitter = RecursiveCharacterTextSplitter(
      chunk_size = settings.CHUNK_SIZE,
      chunk_overlap = settings.CHUNK_OVERLAP
   )

   return splitter.split_documents(documents)

def make_chunks():
    documents = load_documents()
    return chunk_documents(documents)

if __name__ =='__main__':  
    chunks = make_chunks()
   
    