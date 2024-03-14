from dotenv import load_dotenv
load_dotenv()
import langchain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter  import CharacterTextSplitter
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from redundant_filter import RedundantFilterRetriever

import os 

openai_api_version = "2024-02-15-preview"
deployment_name = os.getenv("EMBED_DEPLOYMENT_NAME")
chat_deployment_name = "java-gpt4-32k"
model_name = "text-embedding-ada-002"
data_loaded=True

langchain.debug=True

text_splitter = CharacterTextSplitter(
    separator="\n", 
    chunk_size=200, 
    chunk_overlap=100
)
loader = TextLoader("facts.txt")

def embed_docs(data_loaded, doc, embeddings,store_name: str)-> bool: 
    print(f"deployment name: {deployment_name}")
    if not data_loaded:
        save_to_vector_store(embeddings, doc, store_name)
    # print(docs[0])
    # embedding = embeddings.embed_query(docs[0].page_content)
    return True

def save_to_vector_store(embeddings, docs, db_name: str): 
    Chroma.from_documents(
        embedding=embeddings,
        documents=docs,
        persist_directory=db_name
    )
    return

def search_similar(embeddings: AzureOpenAIEmbeddings,chat: AzureChatOpenAI, store: str,query: str): 
    db = Chroma(
        persist_directory=store, 
        embedding_function=embeddings
    )
    # retriever is glue interface that interfaces between a specific vector database 
    # and the retrieval QA chain. It is responsible for converting the input query into
    # a vector and performing the similarity search
    # retriever = db.as_retriever()
    retriever = RedundantFilterRetriever(
        embeddings=embeddings, 
        vector_db=db
    ) 
  
    chain = RetrievalQA.from_chain_type(
          llm=chat, 
          retriever=retriever,
          chain_type="stuff" # the stuff chain type just takes the input from vector database and stuffs it into the prompt
    )
    results = chain.invoke(query, top_k=2)
    return results

if __name__ == "__main__":
    store = "emb"
    docs = loader.load_and_split(text_splitter)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=deployment_name,
        openai_api_version=openai_api_version,
        model=model_name
    )
    chat = AzureChatOpenAI(
        deployment_name=chat_deployment_name,
        openai_api_version=openai_api_version,
    )

    data_loaded = embed_docs(data_loaded, docs,embeddings,store_name=store)
    results = search_similar(embeddings,chat, store,"What is an interesting fact about the english language")
    print(f"here is an interesting fact about the english language: {results}")
