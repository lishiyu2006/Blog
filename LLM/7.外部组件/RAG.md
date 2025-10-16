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