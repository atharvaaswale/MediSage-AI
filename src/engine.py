import os
from dotenv import load_dotenv

# Modern LangChain Imports for v1.x
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# For LangChain 1.x, these functions are in the classic package
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
def get_rag_chain():
    """
    Wrap your existing logic here so other files can import the 'brain'.
    """
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    DB_FAISS_PATH = "vectorstore/db_faiss"

    system_prompt = (
        "Use the following pieces of retrieved context to answer the user's medical query. "
        "If you don't know the answer, say you don't know. Do not make up information. "
        "\n\n"
        "Requirements:\n"
        "- Provide professionally and medically sound information.\n"
        "- Use layman-friendly language (explain jargon).\n"
        "- No small talk.\n"
        "- Always include a disclaimer: 'This is for informational purposes and not a substitute for professional medical advice.'\n\n"
        "Context: {context}"
    )

    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 2. Load the Vector Store
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # 3. Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Note: Changed to 1.5-flash as 2.5 doesn't exist yet
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )

    # 1. Contextualize Question Prompt
    # This sub-chain re-writes the user query to include context from chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # This creates a retriever that understands history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 4. Create the Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# for testing locally, won't run in Streamlit
# if __name__ == "__main__":
#     chain = get_rag_chain()
#     user_query = input("Enter your medical query: ")
#     response = chain.invoke({"input": user_query})
#     print(response["answer"])
