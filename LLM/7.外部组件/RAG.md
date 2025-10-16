# RAG
RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与语言生成的技术，常用于提升大语言模型（LLM）在特定领域或知识密集型任务中的准确性与相关性。下面我将带你一步步实操一个简单的 RAG 系统。
## 🧰 一、RAG 基本原理

RAG = **检索器（Retriever）** + **生成器（Generator）**

1. **用户提问** →
2. **检索器** 从知识库中检索相关文档片段（如 PDF、网页、数据库等）→
3. **将问题 + 检索到的上下文** 输入给 LLM →
4. **LLM 生成答案**（基于检索内容，避免“幻觉”）

---

## 🛠️ 二、实操环境准备

我们将使用 Python + 开源工具搭建一个本地 RAG 系统。

### 所需库：

```bash
pip install langchain
pip install faiss-cpu  # 向量数据库（也可用 chroma、weaviate 等）
pip install sentence-transformers  # 用于嵌入（Embedding）
pip install transformers  # 可选，用于本地 LLM
pip install pypdf  # 如果要读取 PDF
```
> 💡 你也可以使用 OpenAI API 作为 LLM，但这里我们尽量用开源方案。

---

## 📚 三、准备知识库（示例）

假设我们有一个本地知识库：`data/` 目录下有几个 `.txt` 文件，内容是关于“人工智能发展史”的片段。

例如：`ai_history.txt`
```
人工智能（AI）起源于20世纪50年代。1956年达特茅斯会议被认为是AI的诞生标志。
...
```
## 🔍 四、构建 RAG 系统（代码）

### 步骤 1：加载文档
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载文档
loader = TextLoader("data/ai_history.txt", encoding="utf-8")
documents = loader.load()

# 分块（chunk）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
```
### 步骤 2：创建向量数据库（使用 FAISS）
```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 使用开源嵌入模型（如 all-MiniLM-L6-v2）
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 构建向量库
vectorstore = FAISS.from_documents(chunks, embeddings)
```
### 步骤 3：设置检索器
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 返回最相关的3个片
```
### 步骤 4：选择 LLM（这里用 HuggingFace 的本地模型）
```python

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 使用较小的开源模型（如 google/flan-t5-base）
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.0,
)

llm = HuggingFacePipeline(pipeline=pipe)
```
> ⚠️ 注意：`flan-t5-base` 是 encoder-decoder 模型，适合问答；若用 LLaMA 等 decoder-only 模型，需调整 pipeline。

### 步骤 5：构建 RAG 链


```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 将所有检索结果拼接后输入 LLM
    retriever=retriever,
    return_source_documents=True
)
```
### 步骤 6：提问并查看结果
```python
query = "人工智能是在哪一年正式提出的？"
result = qa_chain({"query": query})

print("回答:", result["result"])
print("参考来源:")
for doc in result["source_documents"]:
    print("-", doc.page_content[:100] + "...")
```
## 🌐 五、进阶建议

1. **使用更强大的 LLM**：如 LLaMA-2、ChatGLM、Qwen（需 GPU）
2. **使用 Chroma 或 Weaviate** 替代 FAISS，支持持久化
3. **加入 reranker**（如 Cohere Rerank 或 BAAI/bge-reranker）提升检索质量
4. **部署为 Web 应用**：用 Gradio 或 Streamlit 快速搭建 UI
---

## 🧪 六、完整示例（简化版）

你也可以用 LangChain + OpenAI 快速体验（需 API Key）：

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# 假设已有 vectorstore
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
print(qa.run("人工智能起源于哪一年？"))
```