# RAG
RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与语言生成的技术，常用于提升大语言模型（LLM）在特定领域或知识密集型任务中的准确性与相关性。下面我将带你一步步实操一个简单的 RAG 系统。
## 一、RAG 基本原理

RAG = **检索器（Retriever）** + **生成器（Generator）**

1. **用户提问** →
2. **检索器** 从知识库中检索相关文档片段（如 PDF、网页、数据库等）→
3. **将问题 + 检索到的上下文** 输入给 LLM →
4. **LLM 生成答案**（基于检索内容，避免“幻觉”）

---

## 二、实操环境准备

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

## 三、准备知识库（示例）

假设我们有一个本地知识库：`data/` 目录下有几个 `.txt` 文件，内容是关于“人工智能发展史”的片段。

例如：`ai_history.txt`
```
人工智能（AI）起源于20世纪50年代。1956年达特茅斯会议被认为是AI的诞生标志。
...
```

---

## 四、构建 RAG 系统（代码）

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
#chunk 会被封装成 Document 对象，并作为向量库的基本存储和检索单元。
```
#### 数据库选择：工具库 vs 完整系统

- **FAISS (Facebook AI Similarity Search):** 它是由 Meta 开发的一个**高性能索引库**。它专注于如何在内存中以极快的速度进行向量相似度搜索。它不具备存储原始文档、元数据过滤、用户权限管理或网络接口等功能。
    
- **向量数据库 (如 Milvus, Pinecone):** 它们在底层通常集成了 FAISS 或类似的算法库，但在外层包裹了数据库的功能，如：**持久化存储、CRUD（增删改查）、API 接口、多租户隔离、高可用性**等。
### 常用模型推荐

| 用途       | 模型名称                                                          | 特点                                   |
| -------- | ------------------------------------------------------------- | ------------------------------------ |
| **通用中文** | `BAAI/bge-small-zh-v1.5`                                      | 轻量、高效、中文效果好                          |
| **多语言**  | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 支持50+语言                              |
| **长文本**  | `jina-embeddings-v3`（见知识库 [2]）                                | 支持 **8192 tokens**，支持任务定制（Task LoRA） |
| **英文首选** | `all-MiniLM-L6-v2`                                            | 小巧快速，英文效果佳                           |

> ✅ 中文 RAG 项目强烈推荐 `BAAI/bge-*` 系列（由智源研究院发布），在 MTEB 中文榜单上表现优异。

### 步骤 3：设置检索器
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 返回最相关的3个片
```
> **“片”就是指`Document` 对象**，它包含：
> 
>   - `.page_content`：原始文本内容（即你分块后的 chunk 文本）
>   - `.metadata`：元数据（如来源文件名、页码、分块序号等）
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

---

## 五、进阶建议

1. **使用更强大的 LLM**：如 LLaMA-2、ChatGLM、Qwen（需 GPU）
2. **使用 Chroma 或 Weaviate** 替代 FAISS，支持持久化
3. **加入 reranker**（如 Cohere Rerank 或 BAAI/bge-reranker）提升检索质量
4. **部署为 Web 应用**：用 Gradio 或 Streamlit 快速搭建 UI

---

## 六、完整示例（简化版）

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