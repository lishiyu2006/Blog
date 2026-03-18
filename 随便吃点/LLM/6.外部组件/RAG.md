# RAG
RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与语言生成的技术，常用于提升大语言模型（LLM）在特定领域或知识密集型任务中的准确性与相关性。下面我将带你一步步实操一个简单的 RAG 系统。

## 一、RAG 基本原理

RAG = **检索器（Retriever）** + **生成器（Generator）**

### 1. RAG 基础与核心价值

- **定义**：RAG (Retrieval-Augmented Generation) 通过检索外部知识库增强大模型的生成能力。
- **解决的问题**：
     - 大模型的**幻觉问题**（胡说八道）。
     - 知识的**时效性问题**（训练数据滞后）。
     - 企业**数据隐私**问题。
- **基本流程（Naive RAG）**：
     1. **提取 (Extract)**：从PDF、网页等提取文本。
     2. **索引 (Index)**：切分文本（Chunking），Embedding（向量化），存入向量数据库。
     3. **检索 (Retrieve)**：将用户问题向量化，在数据库中查找相似片段。
     4. **生成 (Generate)**：将检索到的片段作为上下文输入LLM，生成答案。

### 2. RAG 技术进阶 (Advanced RAG & Modular RAG)

课程详细讲解了为了克服朴素RAG的局限性而引入的高级技术，分为三个阶段：

#### A. 预检索阶段 (Pre-Retrieval)

- **查询重写 (Query Rewriting)**
	- **痛点**：用户提问往往很模糊（例如：“它好用吗？”）。
    - **方法**：利用LLM将用户的原始问题改写成语义清晰、包含关键词的查询语句。
    - **代码思路**：Prompt模版 -> "你是一个助手，请把下面这个问题改写得更清晰..." -> LLM生成新Query。
- **多重查询 (Multi-Query / Sub-Question)**
    - **痛点**：一个复杂问题包含多个子意图。
    - **方法**：将一个复杂Query拆解成多个子Query，并行去数据库检索。
    - **例子**：用户问“A公司和B公司的营收对比”，拆解为“A公司营收是多少？”和“B公司营收是多少？”。
- **HyDE (Hypothetical Document Embeddings)**
    - **原理**：假设性文档嵌入。
    - **流程**：
	    1. 先让LLM针对用户问题生成一个**“虚构的答案”**（可能包含幻觉，但语义是对的）。
	    2. 将这个“虚构答案”进行Embedding向量化。
	    3. 用这个向量去知识库里检索**真实的文档**。
    - **效果**：解决了“问题”和“文档”在语义空间距离较远的问题（以答找答）。
- **路由机制 (Router)**
    - **方法**：在检索前增加一个分类器（可以是LLM或逻辑判断）。
    - **逻辑**：根据问题类型（如：摘要类、细节类、对比类），决定调用哪个检索器，或者去哪个知识库（如：去SQL查数据还是去向量库查文档）。

#### B. 检索阶段 (Retrieval) & 索引优化

- **索引策略**：
	- **父子索引 (Small-to-Big / Parent-Child Indexing)**
	    - **痛点**：切片太小，丢失上下文；切片太大，检索不精准（包含噪音）。
	    - **方法**：
	        - 将文档切分成**小切片（Child Chunks）**用于向量化和检索（精准匹配）。
	        - 检索命中后，不直接返回小切片，而是返回它所属的**父文档块（Parent Chunk）**。
	    - **效果**：检索时利用了小切片的精准度，生成时给大模型提供了完整的上下文。
	- **句子窗口检索 (Sentence Window Retrieval)**
	    - **方法**：以“句子”为单位进行Embedding和存储。
	    - **逻辑**：当检索命中某句话时，自动把这句话**前后相邻的几句话（Window）**一起捞出来喂给大模型。
	    - **效果**：保证了语义的连贯性，避免断章取义。
	- **知识图谱索引 (Graph RAG)**
	    - **方法**：利用Neo4j等图数据库，将实体（Entity）和关系（Relation）提取出来构建图谱。
	    - **逻辑**：检索时不仅匹配关键词，还沿着图的边（关系）去寻找关联信息。
	    - **效果**：适合处理复杂推理问题（例如：“A公司的子公司的CEO是谁？”）。
	- **混合搜索 (Hybrid Search)**：
		- **痛点**：向量检索（Dense）对专有名词（如产品型号、人名）匹配效果不如关键词检索（Sparse）。
		- **方法**：同时执行两路检索：
        - **BM25**：基于关键词匹配的稀疏检索。
        - **Embedding**：基于语义向量的稠密检索。
	    - **结果**：将两路检索的结果合并。
	 - **RRF 倒排融合 (Reciprocal Rank Fusion)**
	    - **场景**：当使用了混合检索或多路检索时，如何给结果排序？
	    - **算法**：不依赖具体的相似度分数（因为BM25的分数和向量余弦相似度无法直接比较），而是根据文档在各个列表中的**排名（Rank）**进行加权融合。
	    - **公式**：Score = 1 / (k + rank_i)，排名越靠前，得分越高。

#### C. 后检索阶段 (Post-Retrieval)

- **重排序 (Rerank)**
     - **核心逻辑**：
	    1. 第一轮检索（Retrieve）先粗排，为了速度，召回Top 50或Top 100个片段。
	    2. 第二轮引入**Cross-Encoder模型（Reranker）**，对这50个片段和用户问题进行精细的**逐对打分**。
	    3. 根据Rerank的高精度分数，截取Top 3或Top 5给大模型。
	    - **推荐模型**：BGE-Reranker, Cohere Rerank。
	- **上下文压缩 (Context Compression)**
	- **痛点**：召回的内容太长，浪费Token，且无关信息会干扰大模型。
    - **方法**：使用工具（如LLMLingua）识别并删除prompt中对输出贡献度低的token，或者过滤掉无关的句子，只保留核心信息。

### 3. 代码实战与框架 (LangChain & LlamaIndex)

- **LangChain**：基本的RAG链、多重查询Retriever等。
- **LlamaIndex**：从加载数据、构建索引、配置Retriever到Query Engine的全流程。重点展示了如何通过代码实现**混合检索**和**自定义融合算法**。

### 4. Embedding 模型与微调 (Fine-tuning)

深入到了模型层面，讲解了Embedding的原理及如何优化：

- **原理**：介绍了Word2Vec (CBOW, Skip-gram) 的基本原理，解释了Embedding如何将高维稀疏数据转化为低维稠密向量。
- **评估 (Evaluation)**：
	- 介绍了 **MTEB** (Massive Text Embedding Benchmark) 榜单。
    - 演示了如何使用 InformationRetrievalEvaluator 评估模型的 **MRR (平均倒数排名)** 和 **Hit Rate (命中率)**。
- **微调实战**：
    - **数据准备**：如何构建正负样本对（Anchor, Positive, Negative），以及利用LLM自动生成QA对作为训练数据。
    - **工具**：
	    1. 使用 **LlamaIndex** 的微调引擎进行快速微调。
	    2. 使用 **HuggingFace AutoTrain** 进行更底层的微调操作。
    - **目的**：让通用Embedding模型适应特定垂直领域（如医疗、法律）的专业术语，显著提升检索效果。：

### 5. 重排序技术 (Rerank) —— RAG 的“精修师”

深入讲解了重排序技术的原理、流派及微调方法，这是提升RAG准确率的关键步骤。

- **核心原理**：
    - **Bi-Encoder (双编码器)**：用于检索阶段（Embedding），速度快但精度一般，适合海量数据粗筛。
	- **Cross-Encoder (交叉编码器)**：用于重排序阶段，将Query和Document拼接输入模型，计算相关性得分。精度极高但计算昂贵，仅适用于对Top-K（如前50条）进行精排。
- **技术流派**：
    - **BERT-based**：如 BGE-Reranker，基于Encoder架构，效果稳健。
    - **LLM-based**：利用GPT-4等大模型进行排序，方法包括：
	    - **Pointwise**：让LLM给每个文档打分。
	    - **Listwise**：一次性给LLM一堆文档，让其输出排序结果（受限于Context Window）。
    - **Pairwise**：两两比较（冒泡排序思想），效果好但调用次数多。
    - **ColBERT**：多向量交互模型，兼顾了双编码器的速度和交叉编码器的精度。
- **微调实战**：
    - 使用 Sentence-Transformers 库对Rerank模型进行微调。
    - 了如何基于特定业务数据（正负样本对）优化 BGE-Reranker，使其更懂业务逻辑。

### 6. 向量数据库内核 (Vector DB) —— RAG 的“海马体”

课程极其硬核地拆解了向量数据库的底层算法，这在一般教程中很少见。

- **核心挑战**：在高维空间（High-dimensional space）中进行大规模数据的快速检索。
   - **索引算法 (Indexing Algorithms)**：
     - **FLAT (暴力搜索)**：精度100%，但速度慢，仅适合小数据量。
     - **IVF (倒排文件索引)**：通过聚类（K-Means）将向量空间划分为多个Cell（簇），检索时只搜最近的几个簇，牺牲少量精度换取速度。
     - **HNSW (分层小世界图)**：**目前的SOTA算法**。基于图结构，通过“跳表”思想在不同层级间快速逼近目标，性能与精度极其平衡。
     - **LSH (局部敏感哈希)**：通过Hash函数将相似向量映射到同一个桶，适合超大规模数据。
     - **PQ (乘积量化)**：一种有损压缩技术，将高维向量切分并聚类压缩，大幅降低内存占用（如将100%压缩至3%），但需配合重排序使用。
   - **选型指南**：
     - **专用向量库**：Milvus（功能全、分布式）、Chroma（轻量、本地）、Qdrant（Rust编写、性能好）、Pinecone（云原生）。
     - **传统数据库扩展**：pgvector (PostgreSQL插件)、Elasticsearch (支持向量但在高维性能上弱于专用库)、Redis。

   ### 7. RAG 效果评估 (Evaluation) —— 拒绝“盲目优化”

   如何科学地衡量RAG系统的好坏？课程介绍了一套完整的评估体系。

   - **评估痛点**：肉眼看效果不靠谱，需要量化指标。
   - **评估框架**：
     - **Ragas**：业界主流框架，无需人工标注集（Ground Truth），利用强力LLM（如GPT-4）作为裁判。
     - **TruLens**：另一种流行的评估工具。
   - **核心指标 (RAG Triad)**：
     1. **Context Relevance (上下文相关性)**：检索出来的文档是否真的回答了问题？（评估检索器）
     2. **Groundedness (忠实度/幻觉检测)**：LLM生成的答案是否每一句话都有据可查？（评估生成器）
     3. **Answer Relevance (答案相关性)**：生成的答案是否直接回应了用户的问题？
   - **代码实战**：演示了如何在代码中集成Ragas，生成可视化的评估分数表格，指导系统优化方向。

   ### 8. 行业落地经验与架构设计

   课程最后拔高到了架构师视角，分享了企业落地的真实经验。

   - **技术栈选择**：
     - **Python**：AI算法、模型微调、RAG核心逻辑开发的主流语言。
     - **Java/Go**：由于企业原有系统多为此类语言，通常通过API调用Python封装的AI服务，实现业务逻辑与AI能力的解耦。
     - **Spring AI**：介绍了Java生态接入LLM的方案，适合Java工程师。
   - **RAG vs Fine-tuning (微调) 决策表**：
     - **知识更新频繁** -> 选 **RAG**。
     - **需要特定语气/风格/格式** -> 选 **微调**。
     - **减少幻觉** -> **RAG** 优于微调（因为有来源依据）。
     - **复杂推理** -> 强力底座模型 + **CoT** (思维链) 优于单纯微调。
   - **数据工程 (Data Engineering)**：
     - 强调**数据清洗**的重要性（Garbage In, Garbage Out）。
     - 针对PDF解析、表格提取的难点，推荐了结合OCR的方案。
     - 建议在元数据（Metadata）中加入时间、部门等标签，实现**混合检索（Hybrid Search + Filtering）**。

2. 二、实操环境准备

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