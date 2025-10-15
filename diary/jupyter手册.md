# jupyter手册

jupyter分段式，在web开发的工具
首先Jupter lab是更高级的Jupter notebook，又有他的全部功能；
然后讲介绍什么安装和使用：
## 一、安装
### pip安装：

```shell
pip install jutpyterlab
```

### anaconda安装：

```shell
conda install -cconda-forge jupyterlab
```

这里还有一种就是，用anaconda的应用：

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151635255.png)
## **运行Jupyter Lab**

在安装Jupyter Lab后，接下来要做的是运行它。  
你可以在命令行使用jupyter-lab或jupyter lab命令，然后默认浏览器会自动打开Jupyter Lab。

目录：windows
- **默认情况**：JupyterLab 默认根目录是当前用户的主目录，比如 `C:\Users\你的用户名` 。你可以打开文件资源管理器，在地址栏输入 `%USERPROFILE%` 并回车，就能快速跳转到该目录。
- **查看或修改方法**：通过运行 `jupyter --config-dir` 命令，会得到配置文件所在目录，进入该目录找到 `jupyter_notebook_config.py` 文件（如果没有则可以创建） ，打开后查找或添加以下配置项：
```python
c.NotebookApp.notebook_dir = '你想要设置的路径'
```
例如 `c.NotebookApp.notebook_dir = 'D:/my_jupyter_workspace'` ，保存文件后，重启 JupyterLab，新的根目录设置就会生效。
