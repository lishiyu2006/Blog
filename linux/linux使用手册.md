# Linux 系统目录结构

输入

`/s`

输出有哪些子目录，下面有各个目录的用法

![img](https://www.runoob.com/wp-content/uploads/2014/06/4_20.png)

 树状目录结构

![img](https://www.runoob.com/wp-content/uploads/2014/06/d0c50-linux2bfile2bsystem2bhierarchy.jpg)

- `/bin`：存放常用命令，如`cat`、`chmod`等。

- `/usr/bin`:系统用户使用的应用程序。

- `/sbin`：系统管理程序。

- `/usr/sbin`:超级用户使用的比较高级的管理程序和系统守护程序。

- `/etc`：配置文件保存位置。

- `/var`：可修改的日志文件和邮件预设放置。

  > [!IMPORTANT]
  >
  > 以上不可以随便更改，可能会导致系统不能启动

- `/boot`：存储启动Linux时使用的核心文件。

- `/dev`：Device的缩写，设备文件保存位置。

- `/home`：用户主目录，可自定义，如图上的alice，bob，eve。

- `/lib`：Library(库) 的缩写，动态链接库。

- `/lost+found`：用于非法关机后的临时文件。

- `/media`：自动识别并挂载设备,如例如U盘、光驱等等，将存储在该目录下。

- `/mnt`：临时挂载其他文件系统。

- `/opt`： optional(可选) 的缩写，额外软件存放位置。

- `/proc`：Processes(进程) 的缩写，虚拟文件系统。

- `/root`：管理员主目录。

- `/selinux`：Redhat/CentOS特有的安全机制，类似于windows的防火墙。

- `/srv`：服务启动数据。

- `/sys`：集成多种文件系统信息。

- `/tmp`：临时文件存放位置。

- `/usr`：应用程序和共享资源。

- `/usr/src`:内核源代码默认的放置目录。

接下来我们就来看几个常见的处理目录的命令吧：

- `ls`（list files）: 列出目录及文件名

相关参数：

1. **-a** ：全部的文件，连同隐藏文件( 开头为 . 的文件) 一起列出来(常用)
2. **-d** ：仅列出目录本身，而不是列出目录内的文件数据(常用)
3. **-l** ：长数据串列出，包含文件的属性与权限等等数据；(常用）
4. **-al~**：将目录下的所有文件列出来（含属性和隐藏档）
5. **ls -r**：用于反转文件的显示顺序
6. **ls-R**：递归地列出目录树

- `cd`（change directory）：切换目录

1. **[root@www ~]# mkdir runoob**    #使用 mkdir 命令创建 runoob 目录
2. **[root@www ~]# cd /root/runoob/**    #使用绝对路径切换到 runoob 目录
3. **[root@www ~]# cd ./runoob/**    #使用相对路径切换到 runoob 目录
4. **[root@www runoob]# cd ~**    #表示回到自己的家目录，亦即是 /root 这个目录
5. **[root@www ~]# cd ..**    #表示去到目前的上一级目录，亦即是 /root 的上一级目录的意思；

- `pwd`（print work directory）：显示目前的目录

1. **-P** ：显示出确实的路径，而非使用链接 (link) 路径

2. ```
   [root2www ~]# cd /var/mail  <=这里/var/mail是一个链接档
   [root@www mail]# pwd
   /var/mail
   [root@www mail]# pwd -P
   /var/spool/mail
   ```

   > [!CAUTION]
   >
   > 链接档类似于Windows中的快捷方式

- `mkdir` (make directory）：创建一个新的目录

1. **-m** ：配置文件的权限喔！直接配置，不需要看默认权限 (umask) 的脸色～

2. **-p** ： 嵌套创建文件夹

3. ```
   [root@www tmp]# mkdir test    <==创建一名为 test 的新目录
   [root@www tmp]# mkdir -p test1/test2/test3/test4
   ```

- `rmdir`（remove directory）：删除一个空的目录

1. ```
   [root@www tmp]# rmdir test   <==可直接删除掉，没问题
   [root@www tmp]# rmdir test1  <==因为尚有内容，所以无法删除！
   [root@www tmp]# rmdir -p test1/test2/test3/test4   <== -p 选项，可以将 test1/test2/test3/test4 一次删除
   ```

2. **-p ：**从该目录起，一次删除多级空目录

- `cp`（copy file）: 复制文件或目录

1. **-a：**相当于-pdr 的意思，至于 pdr 请参考下列说明；(常用)
2. **-d：**若来源档为链接档的属性(link file)，则复制链接档属性而非文件本身；
3. **-f：**为强制(force)的意思，若目标文件已经存在且无法开启，则移除后再尝试一次；
4. **-i：**若目标档(destination)已经存在时，在覆盖时会先询问动作的进行(常用)
5. **-l：**进行硬式链接(hard link)的链接档创建，而非复制文件本身；
6. **-p：**连同文件的属性一起复制过去，而非使用默认属性(备份常用)；
7. **-r：**递归持续复制，用於目录的复制行为；(常用)
8. **-s：**复制成为符号链接档 (symbolic link)，亦即『捷径』文件；
9. **-u：**若 destination 比 source 旧才升级 destination ！

- `rm`（remove）: 删除文件或目录

1. -f ：就是 force 的意思，忽略不存在的文件，不会出现警告信息；

2. -i ：互动模式，在删除前会询问使用者是否动作，y，n

3. -r ：递归删除啊！最常用在目录的删除了！这是非常危险的选项！！！

4. ```
   [root@www tmp]# rm -i bashrc
   rm: remove regular file `bashrc'? y
   ```

- `mv`（move file）: 移动文件与目录，或修改文件与目录的名称

1. -f ：同上
2. -i ：同上
3. -u ：若目标文件已经存在，且 source 比较新，才会升级 (update)
- `apt` 命令简洁高效

你可以使用 *man [命令]* 来查看各个命令的使用文档，如 ：man cp。