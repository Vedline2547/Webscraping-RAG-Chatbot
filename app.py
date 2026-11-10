
# Document loaders
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# LLM
from langchain_ollama import ChatOllama

# Prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hugging Face embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS vector database
from langchain_community.vectorstores import FAISS

# Custom prompts
from prompts import qa_system_prompt, contextualize_q_system_prompt

# Chains
from langchain_classic.chains import (
    RetrievalQA,
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

# Chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Flask
from flask import Flask, render_template, request, session

# Other libraries
import os

# CONFIGURATION
FAISS_PATH = "faiss"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "development-secret-key")

# LLM

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    keep_alive="30m"
)

# CONVERSATION HISTORY

conversation_store = {}
display_history = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:

    if session_id not in conversation_store:
        conversation_store[session_id] = ChatMessageHistory()

    return conversation_store[session_id]

# DOCUMENT LOADER

def get_document_loader():

    loader = DirectoryLoader(
        "static",
        glob="**/*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader
    )

    docs = loader.load()

    return docs

# TEXT SPLITTING
def get_text_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

# EMBEDDINGS AND FAISS
def get_embeddings():

    path = os.path.join(os.getcwd(), FAISS_PATH)

    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(path):

        print(f"Loading FAISS index from {path}")

        db = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        print("FAISS index does not exist. Creating one.")

        documents = get_document_loader()

        if not documents:
            raise ValueError("No PDF documents found in the static folder.")

        chunks = get_text_chunks(documents)

        db = FAISS.from_documents(
            chunks,
            embeddings
        )

        db.save_local(path)

        print("FAISS index created successfully.")

    return db



# RETRIEVER

def get_retriever():

    db = get_embeddings()

    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever

# RAG CHAIN

def get_chain():

    retriever = get_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain

# INITIALIZE COMPONENTS ONCE

print("Initializing RAG application...")

retriever = get_retriever()

rag_chain = get_chain()

print("RAG application ready.")

# FLASK ROUTES

@app.route("/")
def index():

    return render_template("home.html")


@app.route("/chat", methods=["GET", "POST"])
def document_display():

    if request.method == "GET":
        return render_template("chat.html")

    question = request.form.get("question", "").strip()

    if not question:
        return render_template(
            "chat.html",
            chat_history=[]
        )

    # Create a session ID for each user
    if "session_id" not in session:
        import uuid
        session["session_id"] = str(uuid.uuid4())

    session_id = session["session_id"]

    # Run the RAG chain
    response = rag_chain.invoke(
        {"input": question},
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    answer = response["answer"]

    # Maintain display history separately
    if session_id not in display_history:
        display_history[session_id] = []

    display_history[session_id].append({
        "question": question,
        "answer": answer
    })

    return render_template(
        "chat.html",
        chat_history=display_history[session_id]
    )


# RUN APPLICATION

if __name__ == "__main__":

    app.run(debug=True,use_reloader=False)
