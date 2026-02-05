# docker
## windows

dcker的软件是默认在c盘的一个特殊的轻量linux里面WSL，在桌面软件docker desktop里面的设置可以看见wsl在哪个位置

如果你在界面上没看到这个选项，或者迁移失败，可以使用 WSL 命令手动迁移：

1. **彻底关闭 Docker**：在右下角托盘退出 Docker Desktop。
    
2. **查看状态**：打开 PowerShell，输入 `wsl -l -v`，确保 `docker-desktop-data` 的状态是 `Stopped`。如果没停止，输入 `wsl --shutdown` 强制关闭。
    
3. **导出数据**：将现有的 Docker 数据导出为一个文件（假设存到 D 盘）：
    
    PowerShell
    
    ```
    wsl --export docker-desktop-data D:\docker-desktop-data.tar
    ```
    
4. **注销原有的数据**：
    
    PowerShell
    
    ```
    wsl --unregister docker-desktop-data
    ```
    
5. **导入到新位置**：将数据导入到你想要存放的目录（例如 `D:\DockerData`）：
    
    PowerShell
    
    ```
    wsl --import docker-desktop-data D:\DockerData D:\docker-desktop-data.tar --version 2
    ```
    
6. **重启 Docker**：现在你可以删掉 `D:\docker-desktop-data.tar` 这个临时文件，重新启动 Docker Desktop 即可。

查看当前镜像

~~~bash
docker images
~~~

停止镜像
~~~bash
docker stop my-chroma
~~~

删除镜像
~~~bash
docker rm my-chroma
~~~



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

检查chroma是否运行，在浏览器输入：`http://localhost:8000/api/v2/heartbeat` 如果你看到一串类似 `{"nanosecond heartbeat": ...}` 的数字，说明它已经活了。