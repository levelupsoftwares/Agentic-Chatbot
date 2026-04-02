from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
# llm + parser

parser = StrOutputParser()
llm = ChatGroq(model_name='openai/gpt-oss-120b')

# load document
loader = TextLoader("data/about_us.txt")
documents = loader.load()   

# split in chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    # separators='',
    chunk_overlap=50
)
splitted_doc = splitter.split_documents(documents)
# print(splitted_doc[23])

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

# vector store
vector_store =Chroma(
    embedding_function=embedding,
    persist_directory='vectorstore',
    collection_name='aboutus'
)

# prevent duplication
if vector_store._collection.count() == 0:
    vector_store.add_documents(splitted_doc)
     


# retriever
retriever = vector_store.as_retriever(
    search_kwargs={'k':5}
)

#formating for retiever object
def formating(docs):
     return "\n".join(doc.page_content for doc in docs)


prompt_template= ChatPromptTemplate([
        ('system',"""You are a helpful assistant.
                    Use ONLY the provided context to answer.
                    If answer is not found in context, reply exactly:
                    "I can only answer questions related to the provided document."
                    Context:{context}"""
        ),
        ('human','{user_query}')
])



chain =({
     # user_query -> retriever -> docs -> formating
     'context':RunnableLambda(lambda x:formating(retriever.invoke(x['user_query']))),

     #passing orginal query
      'user_query':RunnablePassthrough() | RunnableLambda(lambda x:x['user_query'])}
      | prompt_template | llm | parser
) 

while True:
     user_input= input(f'Enter Query: ') 
     if user_input == "exit":
          break 
     print(chain.invoke({'user_query':user_input}))
     