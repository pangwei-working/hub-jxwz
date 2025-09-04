# rag_test.py
### Build Index
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


# 加载环境变量
load_dotenv()

# Set local embedding model
embd = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings":True},    # 归一化向量
    model_kwargs={"device": "cpu"},  # Use CPU for local inference
)

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

print(f"Loaded {len(docs_list)} documents from {len(urls)} URLs.")
# Print first document
print(f"First document: {docs_list[0].page_content[:100]}...")
# Print first document metadata
print(f"First document metadata: {docs_list[0].metadata}")
# Print first document metadata keys
print(f"First document metadata keys: {docs_list[0].metadata.keys()}")
# Print first document metadata values
print(f"First document metadata values: {docs_list[0].metadata.values()}")

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

print(f"Split into {len(doc_splits)} document chunks.")
# Print first document chunk
print(f"First document chunk: {doc_splits[0].page_content[:100]}...")
# Print first document chunk metadata
print(f"First document chunk metadata: {doc_splits[0].metadata}")
# Print first document chunk metadata keys
print(f"First document chunk metadata keys: {doc_splits[0].metadata.keys()}")
# Print first document chunk metadata values
print(f"First document chunk metadata values: {doc_splits[0].metadata.values()}")

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)
retriever = vectorstore.as_retriever()
print(f"Vectorstore created with {len(doc_splits)} document chunks.")
# Save vectorstore
vectorstore.persist()
print("Vectorstore persisted.")
print(f"Vectorstore is saved in the default Chroma directory: './chroma_db' (relative to your current working directory).")