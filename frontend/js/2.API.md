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

- **BaseURL (基础地址)**: `https://api.weather.com/v1`
    
- **Endpoint (端点)**: `/current`
    
- **Method (方法)**: `GET`（表示获取数据）
    
- **Parameters (参数)**: `city=Shanghai`, `unit=c` (摄氏度), `apikey=12345`

2. api响应

接收端在接到这个请求之后,会检查apikey(通行证)是否有效。没问题就返回一串 JSON 格式的数据

~~~js
{
  "status": "success",
  "data": {
    "city": "Shanghai",
    "temperature": 22,
    "condition": "Partly Cloudy",
    "humidity": "65%",
    "update_time": "2026-03-24 17:30"
  }
}
~~~