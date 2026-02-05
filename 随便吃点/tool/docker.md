# docker
## windows

### 展示正在运行的容器

~~~bash
docker ps
~~~

- **`docker ps -a`** (all): 列出**所有**容器，包括正在运行的和已经停止（Exited）的。如果你发现容器启动失败了，用这个命令才能看到它。
    
- **`docker ps -q`** (quiet): 只显示容器的 ID。这在写脚本或者批量操作（比如删除所有容器）时非常有用。
    
- **`docker ps -s`** (size): 除了显示基本信息，还会显示容器占用的磁盘大小。
    
- **`docker ps -f`** (filter): 按条件过滤，比如 `docker ps -f "status=exited"` 只看停止运行的容器。

### dcker的软件是默认在c盘的一个特殊的轻量linux里面WSL，在桌面软件docker desktop里面的设置可以看见wsl在哪个位置

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

### 查看当前镜像

~~~bash
docker images
~~~

### 停止镜像
~~~bash
docker stop my-chroma
~~~

### 删除镜像
~~~bash
docker rm my-chroma
~~~

### 让容器运行某个端口上

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

### ”管家“一次性加载=多次run

例：dify的开发者提前写好了，一个.yaml文件，运行他

~~~bash
docker-compose up -d
~~~

Docker Compose 会在**当前文件夹**下疯狂寻找一个名为 **`docker-compose.yaml`**（或 `.yml`）的文件。

除此之外，还有

- **`docker-compose stop`**：
    
    - **意思**：暂停所有正在运行的服务，但不会删除它们。
        
- **`docker-compose down`**：
    
    - **意思**：停止并**移除**所有容器和网络，清理现场。

`docker-compose.yaml` 文件的结构通常如下：

~~~yaml
services:
  web:           # 前端服务
    image: nginx
  db:            # 数据库服务
    image: postgres
  redis:         # 缓存服务
    image: redis
~~~