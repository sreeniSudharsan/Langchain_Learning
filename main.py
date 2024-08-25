import bs4
import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore


load_dotenv()

llm = ChatCohere(model = "command-r-plus")

loader = WebBaseLoader(
    web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent",),
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)
vector_store = Chroma.from_documents(documents = splits, embedding= CohereEmbeddings(model = "embed-english-v3.0"))
retriever = vector_store.as_retriever()

system_prompt = (
    "You are an assistant for a question answering task. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you do not know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer = create_stuff_documents_chain(llm, prompt)
rag = create_retrieval_chain(retriever, question_answer)

response = rag.invoke({"input": "What is task decomposition?"})
print(response["answer"])
