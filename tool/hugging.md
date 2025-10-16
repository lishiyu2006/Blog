# Hugging
## 一、 Hugging Face简介
Hugging Face（抱脸网）是一个知名的开源库和平台，该平台以其强大的Transformer模型库和易用的API而闻名，为开发者和研究人员提供了丰富的预训练模型、工具和资源。

以下是对 Hugging Face 平台的简要介绍：

- 预训练模型库：Hugging Face提供了大量先进的预训练模型，起初涵盖了各种NLP任务，如文本分类、命名实体识别、情感分析、问答等。随着平台的壮大，如今各个领域的预训练模型几乎都可以在平台上找到，这让我们可以关注研究问题本身，而不是从论文复现开始“造轮子”。
- transformers 库和diffusers库：Hugging Face的transformers库和diffusers库提供了简洁易用的API，使得下载、加载和使用预训练模型变得非常简单。transformers库中提供的多种分词器和文本编码器为我们研究NLP任务提供了极大的遍历；diffusers库提供了包括文生图(stable diffusion)、文生视频(stable video diffusion)等模型，无论是模型的代码实现还是模型的预训练权重都为我们的学习和使用带来巨大便利。
- 模型架构的创新：Hugging Face不断推出新的模型架构，以改进各种NLP任务的性能。例如，BERT、Llama、RoBERTa、SD、SVD等模型在各自领域取得了巨大成功，并为社区提供了许多基于这些模型的解决方案。
- 社区和开放性：Hugging Face 是一个活跃的社区，拥有庞大的用户群体和开发者社区。平台鼓励用户贡献自己的模型、代码和工具，并提供了开放的 API 和数据集，以促进合作和创新。

## 二、 以Stable Diffusion v1.5为例的基本使用
1. 进来之后界面是这样的，左边是个人的一些东西，中间会展示你follow的一些作者，右边展示比较火热的东西，可以说除了上面的搜索框基本都没啥用
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161413541.png)
2. 接下来，在输入框输入想要的模型就ok了，这里以Stable Diffusion v1.5为例，所以输入框输入stable-diffusion-v1-5/stable-diffusion-v1-5，按下“enter”，成功跳转。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161413332.png)
然后就可以跟着教程下载了
## 三、更优雅的Hugging Face模型下载方式
上面的下载方式很方便，但是存在下载速度慢、可读性差、受网络波动影响大等问题。因此本节讲授本推文最核心的内容：优雅地使用命令行下载Hugging Face模型。

1. 首先安装huggingface-hub库：huggingface-hub，这个库能让我们调用Hugging Face提供的一些API来完成任务。我认为最大的意义就是将我们的精力从Hugging Face网站转移到目前处理的任务上面。可以通过命令huggingface-cli --help验证是否安装成功。
2. 在命令行输入huggingface-cli login，用来通过命令行接入Hugging Face。之后会要求我们输入一个Hugging Face的token，我们访问给出的网址，点击右上角的“Create new token”，只需要再token name输入框中填上你喜欢的名字就可以了，翻到页面最下面点击“Create token”按钮完成token创建。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161415111.png)
这个时候退出去你就会发现自己找不到密钥了
右边三个点的下拉菜单有“Invalidate and refresh”按钮，弹窗点击“确定”按钮就行，你就能再复制token了
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161415589.png)
3. ctrl-c/v输入token，再输入Y，然后就能看到登录成功login successful（红字不重要）。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161416758.png)
4. 执行下载命令：
```bash
huggingface-cli download --resume-download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir /dir_path/hugging-face-models/sd1.5
```
 注意修改保存的地址
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161417983.png)
5. 最后，调用下载好的模型就可以了，把之前的model_id更换为你下载模型的本地路径。至于为什么能这样替换，如果你点进from_pretrained看看方法定义，你会发现第一个参数名字叫“pretrained_model_name_or_path”
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161419161.png)
