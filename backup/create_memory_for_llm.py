from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Step1: load raw pdf

DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader,
        silent_errors=True,# Skip broken PDFs without crashing
        show_progress=True
    )
    documents = list(loader.lazy_load())
    print(f"Loaded {len(documents)} PDF pages/documents.")
    return documents

documents = load_pdf_files(DATA_PATH)

# Step2: Creaete chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", "!", "?", " ", ""])
    # doc.page_content is the text content of each page in the PDF, doc in documents is a list of Document objects, 
    # and we are extracting the text content from each Document object to create chunks.
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    return chunks

text_chunks = create_chunks(documents)


# Step3: Create Vector Embeddings
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


embeddings = get_embeddings()


# Step4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorestore/faiss_db"
db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)

print("Vector store successfully saved to: ", DB_FAISS_PATH)