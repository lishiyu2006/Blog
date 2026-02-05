# docker
## windows

让某个东西运行某个端口上

例：下载运行chroma

~~~bash
docker run -d `     # 1. 在后台运行，别把命令行窗口占用了
  --name my-chroma `    # 2. 给这个容器起个名字叫 my-chroma
  -p 8000:8000 `     # 3. 门牌号映射：把容器的 8000 端口接到 Windows 的 8000 端口
  -v E:\software\ai-project\dify-agent `   # 4. 关键！把 D 盘这个文件夹当成数据库的硬盘
  --restart always `   # 5. 告诉 Docker：去下载并运行这个名为 chroma 的程序
  chromadb/chroma         
~~~

- **`-d`**: 后台运行。
    
- **`-p 8000:8000`**: 把容器的 8000 端口映射到你 Windows 的 8000 端口。
    
- **`-v c:\chroma_data:/chroma/chroma`**: **非常重要！** 这把数据存在你电脑的 
    
- `E:\software\ai-project\dify-agent` 文件夹里。即使你删了容器，售后知识库的数据也不会丢。
    
- **`--restart always`**: 保证你重启电脑后，Chroma 会自动启动。