from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain_core.documents import Document
from pathlib import Path



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

def load_documents():
    docs_total = []
    loader = DirectoryLoader(
        path=Path(__file__).parents[2]/'data/raw',
        glob='**/*.txt',
        loader_cls=TextLoader,
        recursive=True
    )


    for page in loader.lazy_load():
        content = page.page_content
        file_path = page.metadata["source"]
        docs_total.append(
            Document(
                page_content = content,
                metadata = {
                    **page.metadata,
                    **get_metadata(file_path)}
                )
        )

    return docs_total
