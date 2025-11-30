# conda
## 两种conda
### Minconda and Anaconda

Anaconda是一个包含了conda、Python和超过150个科学包及其依赖项的科学Python发行版，预先包含了大量的库，如NumPy, Pandas, Scipy, Matplotlib等。

相较之下，Miniconda更加轻量级。它只包含了Python和Conda。Miniconda用户需要手动安装他们需要的包，这使得Miniconda的环境更为简洁，可以根据实际需求来安装必要的包，避免不必要的存储占用。

## 基本操作

看环境
~~~bash
conda env list   #看有什么环境
conda list    #看当前环境的包
~~~

换环境
~~~bash
conda actviate your_env_name
~~~

安装
~~~
conda install "Url"
~~~

卸载
~~~
conda remove "name"
~~~







