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
## 二、**运行Jupyter Lab**

在安装Jupyter Lab后，接下来要做的是运行它(软件版安装以后默认在c盘打开，建议终端安装，手动输入地址打开)。

打开：
```shell
jupyter-lab
# or jupyter lab
```

关闭：
1. 关闭浏览器中的 Jupyter Lab 页面；
2. 回到命令行窗口，按 `Ctrl+C` 两次，或输入 `y` 确认关闭服务，命令行显示 `Shutting down server` 即表示成功关闭。


目录：windows
- **默认情况**：JupyterLab 默认根目录是当前用户的主目录，比如 `C:\Users\你的用户名` 。
- **查看或修改方法**：通过运行 `jupyter --config-dir` 命令，会得到配置文件所在目录，进入该目录找到 `jupyter_notebook_config.py` 文件（如果没有则可以创建） ，打开后查找或添加以下配置项：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151719829.png)

Ctrl+F搜索找到c.ServerApp.notebook_dir = ''这一行
```python
# c.ServerApp.notebook_dir =''
```
改成
```python
c.ServerApp.notebook_dir ='你想要设置的路径'
```
例如 `c.NotebookApp.notebook_dir = 'D:/my_jupyter_workspace'` ，保存文件后，重启 JupyterLab，新的根目录设置就会生效。

注意：  
1.前面的#必须去掉，不然不生效  
2.单引号里写路径，有的说要写双斜杠，实测单斜杠就可以  
3.有说设置c.ServerApp.root_dir的，这个不要设置，我设置完了之后c.ServerApp.notebook_dir 就失效了，工作目录又变回默认的用户文件夹下面了。
4.发现有个问题，这样设置在命令行中启动juputer notebook没问题，但是通过快捷方式启动还是不行，需要修改快捷方式的属性：

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151721202.png)

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151740291.png)

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151740520.png)

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151740565.png)

## 三、解决jupyter没法识别anaconda的环境

在本地就可以识别有cuda的GPU，在jupyter就不行，为什么呢？
因为PyCharm里添加了对应的解释器（安装了CUDA PyTorch的虚拟环境），但是在jupyter notebook里使用的仍然是普通的python解释器，没有应用虚拟环境

### 解决方法：

#### 要进行以下操作，终端进行以下操作：

1. 检测是否安装jupyter：

因为就算没有这个依旧可以正常使用，Jupyter 本身可以安装在一个环境里（比如 base 环境），但内核可以注册来自其他环境（比如 `new_torch_env`）。所以即使 `new_torch_env` 里没装 Jupyter，只要 base 环境装了 Jupyter，也能启动它，只是需要把 `new_torch_env` 的内核正确注册进去

```bash
jupyter --version
```

显示：

```
jupyter core : 4.12.0 jupyter-notebook : 6.5.4 ipykernel : 6.23.1 ...（其他组件版本）
```

没有就安装（这三个主要的库）

```bash
conda install jupyter notebook ipykernel -y
```

然后，将本地环境导入到jupyter里面

```python
python -m ipykernel install --name your_env_name
```

删除

```shell
jupyter kernelspec remove new_torch_env
```

检查正确将环境加入到jupyter里面

Installed kernelspec jupyer_name  in D:\your_address

在这里-name后面，是jupyter内核的名字，查找方法：

```bash
jupyter kernelspec list
```

有的话，就是好了，再打开jupyter

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202511041333911.png)

就会出现新的环境了；

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202511041333330.png)
