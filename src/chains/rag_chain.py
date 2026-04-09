from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.pipelines.retrieval_pipline import hybrid_search
from config.settings import settings
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# chat_history = []
llm = ChatGroq(model=settings.LLM_MODEL)
retriever = hybrid_search()
parser = StrOutputParser()


# formater function
def formate_docs(data):
    return "\n".join([page.page_content for page in data])

chain = (
    RunnableParallel({
        'query':RunnablePassthrough(),
        'context':retriever | formate_docs
    })
    |ChatPromptTemplate.from_messages([
        ('system', settings.SYSTEM_PROMPT),
        ('system',"Context:\n{context}"),
        ('human',"{query}")
    ])
    |llm
    |parser
)

print(chain.invoke('tell me about the pricing you charge for the websites'))

# def prompt_assembled(input_query):
#      system_message = settings.SYSTEM_PROMPT
#      query = input_query['query']
#      context = input_query['context']
     
#      final_prompt = ChatPromptTemplate.from_messages([
#          {'system':system_message},
#          {'system':f"context:{context}"},
#          {'human':query}
#      ])
#      prompt_formated = final_prompt.invoke({
#         "system_message":system_message,
#         "context":context,
#         "query":query
#      })
#      return prompt_formated
        


# chain =( RunnableParallel({
#          'query':RunnablePassthrough(),
#          'context':(retriever|RunnableLambda(formater))
#          })
#         |RunnableLambda(prompt_assembled)
#         |llm
# )
# print(chain.invoke('what services are you offering'))

# print(prompt_assembled('iii','context'))
# chain = RunnableLambda(prompt_assembled)
# print(chain.invoke('what your company do')) 

# def context(): 
#      retriever_chain = retriever |RunnableLambda(formater)
#      return retriever_chain



# def rag_chat(context):
#     chat_tempalte = ChatPromptTemplate([
#         ('system',settings.SYSTEM_PROMPT),
#         MessagesPlaceholder(variable_name='context'),
#         # MessagesPlaceholder(variable_name='chat_history'),
#         ('human',"i want to contact how i do")
#     ])
# print(rag_chat(context()))