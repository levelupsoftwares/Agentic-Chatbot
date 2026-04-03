from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain_core.documents import Document
from pathlib import Path

docs_total = []

# function for extracting some lines for the purpose of metadata from the txt file 
def get_metadata(file):
    metadata = {}

    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            if  ':' not in line:
                continue
            key_name,value = line.split(':',1) #seprator + maxSplit

            #store the clean data
            metadata[key_name.strip()]= value.strip()

            if key_name.strip() == 'SOURCE_URL':
                break

    return metadata


loader = DirectoryLoader(
    path=Path(__file__).parents[2]/'data/raw',
    glob='**/*.txt',
    loader_cls=TextLoader,
    recursive=True
)
docs = loader.lazy_load() 

for page in docs:
    content = page.page_content
    file_path = page.metadata["source"]
    docs_total.append(
        Document(
            page_content = content,
            metadata = get_metadata(file_path)
            )        
    )


print(len(docs_total))
