# 设计ai图片生成

## 前后端

- 使用语言：Vue，ts，框架：Nuxt，element
- 前端实现功能：路由（有home，generate）

## 功能

## 图片加上文字

## 图片生成特定效果，内置提示词


 - “前端直传 COS ➔ 后端拿 URL 调生图 API ➔ 生成结果存入 COS ➔ 前端展示”
 - 具体：前端 Element Plus 的 `<el-upload>` 拦截上传，直接把图片推给腾讯云 COS，后端只负责发一个“临时通行证（STS），在 Nuxt 项目的 `server/api/cos-sts.ts` 中，实现一个获取腾讯云临时密钥的接口
 - 数据库连接：腾讯云



