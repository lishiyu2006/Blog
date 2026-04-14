# linux使用指南

## •内核的引导

先BIOS开机自检，按照BIOS中的设置启动设备（一般是通过硬盘）启动

读入“/bool”目录下的内核文件

![img](https://www.runoob.com/wp-content/uploads/2014/06/bg2013081702.png)

## •运行init

init进程是系统所有进程的起点

init 程序首先是需要读取配置文件 /etc/inittab

![img](https://www.runoob.com/wp-content/uploads/2014/06/bg2013081703.png)

### 运行级别

许多程序需要开机启动，在Linux就叫做"守护进程"（daemon）

init进程的一大任务，就是去运行这些开机启动的程序

Linux允许为不同的场合，分配不同的开机启动程序，这就叫做"运行级别"（runlevel）

![img](https://www.runoob.com/wp-content/uploads/2014/06/bg2013081704.png)

- 运行级别0：系统停机状态，系统默认运行级别不能设为0，否则不能正常启动
- 运行级别1：单用户工作状态，root权限，用于系统维护，禁止远程登录
- 运行级别2：多用户状态(没有NFS)
- 运行级别3：完全的多用户状态(有NFS)，登录后进入控制台命令行模式
- 运行级别4：系统未使用，保留
- 运行级别5：X11控制台，登录后进入图形GUI模式
- 运行级别6：系统正常关闭并重启，默认运行级别不能设为6，否则不能正常启动

NFS（网络文件系统）是一种由Sun Microsystems公司于1984年开发的分布式文件系统协议，允许用户通过网络访问远程计算机上的文件，就像访问本地文件一样

## •系统初始化

在init的配置文件中有这么一行： si::sysinit:/etc/rc.d/rc.sysinit　它调用执行了/etc/rc.d/rc.sysinit，而rc.sysinit是一个bash shell的脚本，它主要是完成一些系统初始化的工作，rc.sysinit是每一个运行级别都要首先运行的重要脚本

它主要完成的工作有：激活交换分区，检查磁盘，加载硬件模块以及其它一些需要优先执行任务

/etc/rc.d/rc5.d/中的rc启动脚本通常是K或S开头的连接文件

对于以 S 开头的启动脚本，将以start参数来运行,而如果发现存在相应的脚本也存在K打头的连接，而且已经处于运行态了(以/var/lock/subsys/下的文件作为标志)，则将首先以stop为参数停止这些已经启动了的守护进程，然后再重新运行(这样做是为了保证是当init改变运行级别时，所有相关的守护进程都将重启)

![img](https://www.runoob.com/wp-content/uploads/2014/06/bg2013081705.png)

shell命令行界面

## •建立终端



## •用户登陆系统

## •Linux关机

正确的关机流程为：sync > shutdown > reboot > halt

```
sync 将数据由内存同步到硬盘中。

shutdown 关机指令，你可以man shutdown 来看一下帮助文档。例如你可以运行如下命令关机：

shutdown –h 10 ‘This server will shutdown after 10 mins’ 这个命令告诉大家，计算机将在10分钟后关机，并且会显示在登陆用户的当前屏幕中。

shutdown –h now 立马关机

shutdown –h 20:25 系统会在今天20:25关机

shutdown –h +10 十分钟后关机

shutdown –r now 系统立马重启

shutdown –r +10 系统十分钟后重启

reboot 就是重启，等同于 shutdown –r now

halt 关闭系统，等同于shutdown –h now 和 poweroff
```

最后总结一下，不管是重启系统还是关闭系统，首先要运行 **sync** 命令，把内存中的数据写到磁盘中

关机的命令有 **shutdown –h now halt poweroff** 和 **init 0** , 重启系统的命令有 **shutdown –r now reboot init 6**。

