# Libraries
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

# setting up streamlit titles and input
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload pdf's and chat with their content")
session_id = st.text_input(label = "Session ID",
                           value="Default_session"
)
if 'store' not in st.session_state:
  st.session_state.store = {}
uploaded_files = st.file_uploader(
    label = "Upload PDF files for refrence",
    type = "pdf",
    accept_multiple_files = True
)

if uploaded_files:
  def get_session_history(session_id):
    if session_id not in st.session_state.store:
      st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

  user_input = st.text_input("Your question ")

# keys
groq_api_key = st.secrets["GROQ_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]

# Models
llm = ChatGroq(
    model = "Gemma2-9b-It",
    groq_api_key = groq_api_key
)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Prcessing uploaded pdf
if uploaded_files:
  documents = []
  for pdf_file in uploaded_files:
    temppdf = f"./temp.pdf"
    with open(temppdf, "wb") as file:
      file.write(pdf_file.getvalue())
      file_name = pdf_file.name

    loader = PyPDFLoader(temppdf)
    docs = loader.load()
    documents.extend(docs)

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 5000,
      chunk_overlap = 500
  )
  splits = text_splitter.split_documents(documents)
  vectorstore = FAISS.from_documents(
      documents = splits,
      embeddings = embeddings
  )
  retriever = vectorstore.as_retriever()

# Prompts and chain for history and context
contextualized_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualized_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualized_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

if retriever:
  history_aware_retriever = create_history_aware_retriver(
      llm = llm,
      retriever = retriever,
      prompt = contextualized_q_prompt
  )

# prompts and chain for question answer
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "input")
    ]
)
if history_aware_retriever:
  question_answer_chain = create_stuff_documents_chain(
      llm = llm,
      prompt = qa_prompt
  )
  rag_chain = create_retrieval_chain(
      retriever = history_aware_retriever,
      combine_docs_chain = question_answer_chain
  )
  conversational_rag_chain = RunnableWithChatHistory(
      rag_chain = rag_chain,
      input_message_key = "input",
      history_message_key = "chat_history",
      output_message_key = "answer"
  )

# invoking
if user_input:
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input":user_input},
        config={
            "configurable": {"session_id":session_id},
        }
    )

    st.write(st.session_state.store)
    st.write("Assistant: ", response["answer"])
    st.write("chat history: ", session_history.messages)
