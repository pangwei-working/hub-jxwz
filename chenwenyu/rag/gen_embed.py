import os
import loadenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

os.environ["USER_AGENT"] = "rag_test"

# 初始化 all-MiniLM-L6-v2
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#print("模型支持语言:", model._first_module().tokenizer.get_vocab())  # 主要显示英语词汇

embd = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings":True},    # 归一化向量
    model_kwargs={"device":"cpu"}   # 使用 CPU 进行本地推理)
 )

print(f"Using embedding model: {embd.model_name}")

# 生成单条文本的 embedding
vector = embd.embed_query("Hello world!")
print(vector[:5])  # 打印前5维向量
print(f"Dimension of text: {len(vector)}")    # all-MiniLM-L6-v2 应输出 384

# 或者批量生成文档 embeddings,比较两种方式下Hello World的生成结果是否相同
texts = ["Hello world!", "LangChain is great!"]
vectors = embd.embed_documents(texts)

print(vectors[0][:5])  # 打印第一条向量的前5维
print(vectors[1][:5])  # 打印第二条向量的前5维
print(f"Generated {len(vectors)} document vectors.")

# 计算相似度
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

# 计算英文相似度
text1 = "machine learning"
text2 = "artificial intelligence"
vec1 = embd.embed_query(text1)
vec2 = embd.embed_query(text2)
norm1 = norm(vec1);norm2 = norm(vec2)
print(f"向量范数(应为1.0): {norm1:.4f};{norm2:.4f}")  # 输出 1.0000
similarity = dot(vec1, vec2)
print(f"'{text1}' 和 '{text2}' 的余弦相似度：{similarity:.4f}")

# 计算中文相似度
text1 = "机器学习"
text2 = "人工智能"
vec1 = embd.embed_query(text1)
vec2 = embd.embed_query(text2)
norm1 = norm(vec1);norm2 = norm(vec2)
print(f"向量范数(应为1.0): {norm1:.4f};{norm2:.4f}")  # 输出 1.0000
similarity = dot(vec1, vec2)
print(f"'{text1}' 和 '{text2}' 的余弦相似度：{similarity:.4f}")

# 初始化 araphrase-multilingual-MiniLM-L12-v2

embd = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings":True},    # 归一化向量
    model_kwargs={"device":"cpu"}   # 使用 CPU 进行本地推理)
 )

print(f"Using embedding model: {embd.model_name}")
# 计算英文相似度
text1 = "machine learning"
text2 = "artificial intelligence"
vec1 = embd.embed_query(text1)
vec2 = embd.embed_query(text2)
norm1 = norm(vec1);norm2 = norm(vec2)
print(f"向量范数(应为1.0): {norm1:.4f};{norm2:.4f}")  # 输出 1.0000
similarity = dot(vec1, vec2)
print(f"'{text1}' 和 '{text2}' 的余弦相似度：{similarity:.4f}")

# 计算中文相似度
text1 = "机器学习"
text2 = "人工智能"
vec1 = embd.embed_query(text1)
vec2 = embd.embed_query(text2)
norm1 = norm(vec1);norm2 = norm(vec2)
print(f"向量范数(应为1.0): {norm1:.4f};{norm2:.4f}")  # 输出 1.0000
similarity = dot(vec1, vec2)
print(f"'{text1}' 和 '{text2}' 的余弦相似度：{similarity:.4f}")
