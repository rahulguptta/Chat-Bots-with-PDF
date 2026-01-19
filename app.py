running_on_local = False
STATE = {}

#Libraries required
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# keys
if running_on_local:
  from dotenv import load_dotenv
  load_dotenv(".env",  override = True)
  api_key = os.getenv("GROQ_API_KEY")
  HF_TOKEN = os.getenv("HF_TOKEN")
else:
  api_key = st.secrets["GROQ_API_KEY"]
  HF_TOKEN = st.secrets["HF_TOKEN"]

# Langsmiteh tracking
if running_on_local:
  os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
  os.environ["LANGCHAIN_TRACING_V2"]="true"
  os.environ["LANGCHAIN_PROJECT"]="ChatBots on colab"
else:
  LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
  LANGCHAIN_TRACING_V2 = "true"
  LANGCHAIN_PROJECT ="ChatBots on stremlit"


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions basd on provided context only.
    Please provide the most accurate reponse base on the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embeddings():
  store = STATE if running_on_local else st.session_state
  pdf_dir = "/content/sample_data/research_papers" if running_on_local else "research_papers"

  if "vectors" not in store:
    @st.cache_resource(show_spinner=False)
    def get_embedder():
      emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
      test_vec = emb.embed_query("hello world")
      assert isinstance(test_vec, list) and len(test_vec) > 0, "Embedding model returned empty vector."
      st.write(f"✅ Embedding dim: {len(test_vec)}")
      return emb
    embeddings = get_embedder()

    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    final_documents = text_splitter.split_documents(docs[:50])

    vectors = FAISS.from_documents(final_documents, embeddings)

    store["embeddings"] = embeddings
    store["docs"] = docs
    store["final_documents"] = final_documents
    store["vectors"] = vectors


def load_vector_store():
    store = STATE if running_on_local else st.session_state
    if "vectors" not in store:
        # fallback: build once
        create_vector_embeddings()


def generate_response(user_prompt, engine, api_key, running_on_local = False):
  store = STATE if running_on_local else st.session_state
  llm = ChatGroq(model = engine, groq_api_key = api_key)
  document_chain = create_stuff_documents_chain(llm, prompt)
  load_vector_store()
  
  retriever = store["vectors"].as_retriever()
  print("Vector Database is ready")

  import time
  if user_prompt:
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print(f"Response time : {time.process_time() - start}")
    if running_on_local:
      print(f"bot: {response['answer']}")
      simi_search = input("check similarity search")
      if simi_search.lower() in ['yes', 'true']:
        print("Document similarity serarch")
        for i, doc in enumerate(response['context']):
          print(doc.page_content)
          print('---------------')

    else:
      st.write(response['answer'])
      with st.expander("Document similarity serarch"):
        for i, doc in enumerate(response['context']):
          st.write(doc.page_content)
          st.write('---------------')

# User Input
engine = "llama-3.1-8b-instant"
temperature = 0.7
max_tokens = 150

if running_on_local:
  while True:
    user_prompt = input("user: ").strip()
    if user_prompt.lower() in ['q', 'quit', 'break', 'exit']:
      break
    if not user_prompt:
      continue
    generate_response(
        user_prompt = user_prompt,
        engine = engine,
        api_key = api_key,
        running_on_local = True
    )

else:
  st.title("ChatBots")
  st.sidebar.title("setting")

  engine = st.sidebar.selectbox(
      label = "select a model",
      options = ["llama-3.1-8b-instant"],
      index = 0)
  temperature = st.sidebar.slider(label = "Temperature",
                                  min_value = 0.0,
                                  max_value = 1.0,
                                  value = 0.7)
  max_tokens = st.sidebar.slider(label = "Max Tokens",
                                min_value = 50,
                                max_value = 300,
                                value = 150)


  st.write("Fire a question")
  user_prompt = st.text_input("user:")
  generate_response(
        user_prompt = user_prompt,
        engine = engine,
        api_key = api_key,
        running_on_local = False
    )
