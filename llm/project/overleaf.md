# overleaf

## 2025.11.21 改文章

发现overleaf确实有点东西,下面是一些经常用的小技巧

### 在\author{}里面:
1. 用2个$中间加上{^1,2}表示这个人是第一和二个组织的人,并且是上标表示;
2. 在里面用\orcidA{id号码}表示这个人的orid;
3. * 是表示这个认识通讯,并且不是上标;
4. 在$里面\dag表示 † ,是共一的意思,是上标;

## 2025.11.23
### 公式:
1. $ = \\( 都是公式标记;
2. \\\[ 是表示整行公式;
3. \\cap 是并集
eg.格式:
~~~latex
\begin{equation}
B(g,c) = 
\begin{cases} 
1 & \text{if } U(g,c) > 0, \\
0 & \text{otherwise}.
\end{cases}
\end{equation}
~~~
## 2025.11.25
### figure:
格式:
~~~latex
% --- Figure 1 (Full width / Spanning two columns) ---
\begin{figure}[hbt!]
    \centering
    % Ensure the filename matches exactly (Figure1.eps or Figure1.pdf)
    \includegraphics[width=\linewidth]{Figure1}
    \caption{Overview of the ScAR algorithm and validation framework. (A) Data processing workflow: The gene expression matrix is binarized to compute association metrics via matrix operations. (B) The multi-dimensional validation strategy: ScAR-derived networks are validated against known PPI networks, regulatory networks, and biological pathways. (C) Statistical verification methods used to assess the significance of the association metrics.}
    \label{fig:1} % Label for cross-referencing
\end{figure}
~~~

