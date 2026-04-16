# linux指令

解压
~~~
unzip filename.zip


tar -zxvf filename.tar.gz


tar -jxvf filename.tar.bz2


unrar x filename.rar


7z x filename.7z
~~~

- **如果你想自动覆盖**（不再询问）： 使用 `-o` 参数（overwrite）：
```
 unzip -o filename.zip
```

- **如果你想跳过已存在的文件**（不覆盖）： 使用 `-n` 参数（never overwrite）：
```
  unzip -n filename.zip
```