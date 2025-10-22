# graph

## 库 **NetworkX**

### 应用:

创建一个**无向图（Undirected Graph）**

```python
# 创建空的无向图
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

