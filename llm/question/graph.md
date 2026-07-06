# graph

## 库 **NetworkX**

### 应用:

|维度|NetworkX 图对象|igraph 图对象|
|---|---|---|
|**数据结构底层**|纯 Python 实现，基于字典（`dict`）存储节点和边，灵活性高但效率低。|底层用 C 语言实现，采用更紧凑的内存结构（如数组），内存占用小、运算快。|
|**算法支持**|支持基础算法（如最短路径、连通分量），但复杂算法（如 Leiden 社区检测）需依赖第三方库或自定义实现。|内置大量高级算法（如 Leiden/Louvain 社区检测、谱聚类、最大流等），且优化充分。|
|**属性管理**|节点和边的属性通过字典灵活存储（如 `G.nodes[u]['attr']`），但查询和批量操作效率低。|属性通过向量（`VertexSeq`/`EdgeSeq`）管理，支持批量操作（如 `g_ig.vs['attr']` 直接获取所有节点属性），效率更高。|
|**可视化**|内置简单可视化（依赖 Matplotlib），但交互性差，适合快速预览。|支持高质量静态可视化（内置 Cairo 后端）和自定义样式，且可导出为多种格式（PDF/PNG 等）。|
|**大规模图支持**|处理超过 10 万节点 / 边时速度明显变慢，内存占用大。|可高效处理百万级甚至千万级节点 / 边的图，适合工业级数据。|

创建一个**无向图（Undirected Graph）**

```python
# 创建空的无向图NETworkX对象
G = nx.Graph()

# 添加节点
G.add_node(1)  # 单个节点
G.add_nodes_from([2, 3, 4])  # 批量添加节点

# 添加边
G.add_edge(1, 2)  # 节点 1 和 2 之间的边
G.add_edges_from([(2, 3), (3, 4), (4, 1)])  # 批量添加边

# 查看图的基本信息
print("节点：", G.nodes())  # 输出：节点：[1, 2, 3, 4]
print("边：", G.edges())    # 输出：边：[(1, 2), (2, 3), (3, 4), (4, 1)]
```

检查边函数

```python
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3)])

print(G.has_edge(1, 2))  # 输出：True（存在边 1-2）
print(G.has_edge(1, 3))  # 输出：False（不存在边 1-3）
```

将 **NetworkX 图对象** 转换为 **igraph 图对象**(这个存在的目的是小项目,但是啊要用Leiden算的发检测社区)

```python
import networkx as nx
import igraph as ig

# 1. 用 NetworkX 创建带属性的图
G = nx.Graph()
G.add_node(1, label="A")  # 节点 1 带属性 label="A"
G.add_node(2, label="B")
G.add_edge(1, 2, weight=5)  # 边带权重属性

# 2. 转换为 igraph 图
g_ig = ig.Graph.from_networkx(G)

# 3. 查看转换结果
print("igraph 节点 ID：", g_ig.vs["name"])  # 节点 ID 保留（默认与 NetworkX 一致）
print("节点属性 label：", g_ig.vs["label"])  # 节点属性同步
print("边权重：", g_ig.es["weight"])  # 边属性同步
```

### 使用场景对比

- **优先用 NetworkX 图对象的场景**：
    
    - 快速构建和探索小规模图（如演示、教学、原型开发）。
    - 需要灵活处理非结构化属性（如节点 / 边的多类型标签）。
    - 依赖 Python 生态的其他库（如 Pandas）进行数据联动。

- **优先用 igraph 图对象的场景**：
    
    - 执行复杂算法（如社区检测、大规模网络的中心性计算）。
    - 处理百万级以上节点 / 边的大图（如社交网络、交通网络）。
    - 需要高效的批量属性操作或高性能计算。



# 库PyVis

### 作用

PyVis 是一个用于创建交互式网络可视化的 Python 库，通过该方法可以将 NetworkX 构建的图（如 `nx.Graph()` 创建的无向图）转换为 PyVis 的 `Network` 对象，从而实现网页交互式展示（如节点拖拽、缩放、悬停显示信息等）

```python
import networkx as nx
from pyvis.network import Network

# 1. 用 NetworkX 创建图
nx_graph = nx.Graph()
nx_graph.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4)])  # 添加边

# 2. 转换为 PyVis 图对象
pyvis_graph = Network.from_networkx(nx_graph)

# 3. 保存为 HTML 文件（可在浏览器中打开查看交互式图）
pyvis_graph.show("network.html")
```

