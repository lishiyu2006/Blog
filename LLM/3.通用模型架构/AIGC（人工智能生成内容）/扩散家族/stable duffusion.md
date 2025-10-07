# SD模型（stable duffusion）
Stable Diffusion（简称SD）能够进行文生图（txt2img）和图生图（img2img）等图像生成任务

Stable Diffusion是一个完全开源的项目（模型、代码、训练数据、论文、生态等全部开源），这使得其能快速构建强大繁荣的上下游生态

## 1. Stable Diffusion系列资源

- SD 1.4官方项目：[CompVis/stable-diffusion](https://link.zhihu.com/?target=https%3A//github.com/CompVis/stable-diffusion)
- SD 1.5官方项目：[runwayml/stable-diffusion](https://link.zhihu.com/?target=https%3A//github.com/runwayml/stable-diffusion)
- SD 2.x官方项目：[Stability-AI/stablediffusion](https://link.zhihu.com/?target=https%3A//github.com/Stability-AI/stablediffusion)
- diffusers库中的SD代码pipelines：[diffusers/pipelines/stable_diffusion](https://link.zhihu.com/?target=https%3A//github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion)
- SD核心论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2112.10752)
- SD Turbo技术报告：[adversarial_diffusion_distillation](https://link.zhihu.com/?target=https%3A//static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf)
## **2. 零基础深入浅出理解Stable Diffusion核心基础原理**

### **2.1 零基础理解Stable Diffusion模型工作流程（包含详细图解）**

Stable Diffusion（SD）模型是由Stability AI和LAION等公司共同开发的**生成式模型**，总共有**1B左右的参数量**，可以用于文生图，图生图，图像inpainting，ControlNet控制生成，图像超分等丰富的任务，本节中我们以文生图（txt2img）和图生图（img2img）任务展开对Stable Diffusion模型的工作流程进行通俗的讲解。

**文生图任务是指将一段文本输入到SD模型中**，经过一定的迭代次数，**SD模型输出一张符合输入文本描述的图片**。比如下图中输入了“天堂，巨大的，海滩”，于是SD模型生成了一个美丽沙滩的图片。![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051536876.png)

**而图生图任务在输入本文的基础上，再输入一张图片**，SD模型将根据文本的提示，**将输入图片进行重绘以更加符合文本的描述。比如下图中，SD模型将“海盗船”添加在之前生成的那个美丽的沙滩图片上![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051536846.png)

**那么输入的文本信息如何成为SD模型能够理解的机器数学信息呢？**

很简单，我们需要给SD模型一个**文本信息与机器数据信息之间互相转换的“桥梁”——CLIP Text Encoder模型**。如下图所示，我们使用CLIP Text Encoder模型作为SD模型中的**前置模块**，将输入的文本信息进行编码，生成与文本信息对应的Text Embeddings特征矩阵，再将Text Embeddings用于SD模型中来控制图像的生成：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051538193.png)
