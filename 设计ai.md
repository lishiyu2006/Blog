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



ai-photo-app/
├── src/
│   ├── controllers/      # 业务逻辑（调用腾讯云API、处理生图）
│   │   └── photoController.ts
│   ├── views/            # 前端 HTML 模板 (EJS)
│   │   ├── index.ejs     # 首页（上传照片、选择风格）
│   │   └── result.ejs    # 结果页（展示生成的AI照片）
│   ├── public/           # 静态资源（CSS、JS、本地图片）
│   │   └── css/
│   │       └── style.css
│   └── server.ts         # 后端入口文件
├── .env                  # 腾讯云密钥等环境变量
├── package.json
└── tsconfig.json