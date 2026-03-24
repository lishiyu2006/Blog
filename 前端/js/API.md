# API

## 定义

API 就像是**两个软件系统之间的“传声筒”或“服务员”**。它定义了一套规则，让不同的软件能够互相交流，而不需要知道对方内部是怎么运行的。

一个完整的 API 请求通常由 **BaseURL** + **Path（路径）** 组成：

- **BaseURL**: `https://api.example.com/v1`
    
- **Endpoint**: `/users` 或 `/weather`
    
- **完整地址**: `https://api.example.com/v1/users`

## 使用样例

1. 一个请求的样子

`#https://api.weather.com/v1/current?city=Shanghai&unit=c&apikey=12345`