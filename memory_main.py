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
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
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

#Main Prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Sub Prompt you would include for context. This is used for the chat history object
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.messages import AIMessage, HumanMessage

# chat_history = []

# question = "What is Task Decomposition?"
# # ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

# chat_history.extend(
#     [
#         HumanMessage(content=question),
#         AIMessage(content=ai_msg_1["answer"]),
#     ]
# )

# second_question = "What are common ways of doing it?"
# ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

# print(ai_msg_2["answer"])



store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ai_message_3 = conversational_rag_chain.invoke({
#     "input": "What is task decomposition?"
# }, config= {
#  "configurable": {"session_id": "abc123"}
# })["answer"]

# ai_message_4 = conversational_rag_chain.invoke({
#     "input": "What are the uses for the same?"
# }, config = {
#     "configurable":{"session_id": "abc123"}
# })["answer"]


# for message in store["abc123"].messages:
#     if isinstance(message, AIMessage):
#         prefix = "AI"
#     else:
#         prefix = "User"
#     print(f"{prefix}: {message.content}\n:")

from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(retriever, "blog_post_retriever", "Searches and returns excerpts from Autonomous Agents blog post")
tools = [tool]

print(tool.invoke("task decomposition"))
