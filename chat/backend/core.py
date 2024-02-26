#pip install pexpect

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from typing import Any, Dict, List, Tuple
from langchain.chains import ConversationalRetrievalChain #keeps track of history
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from pinecone import Pinecone


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Tuple[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm( #this is the function that exists "from_llm" instead of from_chaintype when using ConversationalRetrievalChain
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True #keeps track of history
    )
    return qa.invoke({"question": query, "chat_history": chat_history})
