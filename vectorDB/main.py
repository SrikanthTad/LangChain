import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone

pinecone.init(
    api_key="",
    environment="northamerica-northeast1-gcp",
)

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader(
        "/Users/Srikanth/Desktop/intro-to-vector-db/mediumblogs/mediumblog1.txt"
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) #may have to play around with this
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index" #this is from pinecone setup. Euclidiean distance select and embedding dimension is 1536 found on embedding output dimension website.
    )

    qa = VectorDBQA.from_chain_type( #VectorDBQA is replaced with RetrievalQA
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True #stuff means we're just using raw vector store as contexts
    )
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)
