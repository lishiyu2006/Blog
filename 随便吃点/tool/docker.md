# docker
## windows

让某个东西运行某个端口上

例：下载运行chroma

~~~bash
docker run -d `
  --name my-chroma `
  -p 8000:8000 `
  -v c:\chroma_data:/chroma/chroma `
  --restart always `
  chromadb/chroma
~~~

- **`-d`**: 后台运行。
    
- **`-p 8000:8000`**: 把容器的 8000 端口映射到你 Windows 的 8000 端口。
    
- **`-v c:\chroma_data:/chroma/chroma`**: **非常重要！** 这把数据存在你电脑的 
    
- `C:\chroma_data` 文件夹里。即使你删了容器，售后知识库的数据也不会丢。
    
- **`--restart always`**: 保证你重启电脑后，Chroma 会自动启动。