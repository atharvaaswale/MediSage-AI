import os
from dotenv import load_dotenv
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


# 1. Configuration & Environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

# 2. Define the System Prompt
# In the new API, we use a structured ChatPromptTemplate
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

def main():
    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 2. Load the Vector Store
    if not os.path.exists(DB_FAISS_PATH):
        print(f"Error: Vector store not found at {DB_FAISS_PATH}.")
        return

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # 3. Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )

    # 4. Create the Chain (The Modern Way)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # This chain handles combining the retrieved documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # This chain links the retriever to the question_answer_chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. Execution
    user_query = input("Enter your medical query: ")
    
    # In the new API, we use "input" instead of "query"
    response = rag_chain.invoke({"input": user_query})

    # 6. Output Management
    print("\n" + "="*30)
    print("MedSage AI Response:")
    # The key is now "answer" instead of "result"
    print(response["answer"])
    print("="*30)
    
    # Accessing sources is now easier:
    # print("\nSources:", [doc.metadata.get('source') for doc in response["context"]])

if __name__ == "__main__":
    main()
