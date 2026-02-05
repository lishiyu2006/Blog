初始化仓库
~~~bash
git init
~~~
把文件上传到暂存区
~~~bash
git add 文件名（. 表示全部文件）
~~~
提交
~~~bash
git commit -m "上传的名字" #-m 表示提交备注，即后面的名字
~~~
关联仓库
~~~bash
git remote add origin https://github.com/【你的GitHub用户名】/【你创建的仓库名】.git
~~~
看关联了什么仓库
~~~bash
git remote
~~~
看仓库的信息
~~~bash
git remot show 【仓库名】
~~~
删除仓库
~~~bash
git remote rm 【仓库名】 # 删除名为origin的远程关联
~~~
推送到github上
~~~bash
git push # 新版GitHub默认分支是main，旧版是master，若报错就加上git push -u origin master
~~~

例下载dify
~~~bash
d:
# 克隆代码仓库
git clone https://github.com/langgenius/dify.git
# 进入 docker 部署目录
cd dify/docker
~~~