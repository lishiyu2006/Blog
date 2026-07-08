# 设计ai
### 一、 核心技术栈与功能设计

#### 1. 核心技术栈

- **前台框架**：Nuxt 3（基于 Vue 3 + TypeScript 的全栈路由与渲染框架）。
    
- **UI 组件库**：Element Plus（用于快速构建表单、按钮、卡片和上传区域）。
    
- **云存储**：腾讯云 COS（对象存储）+ 数据万象 CI（负责图片实时加字/水印）。
    
- **后端引擎**：Nuxt 3 自带的 Nitro 服务器引擎（负责写核心的 API 接口，无需额外搭建 Express）。
    
- **数据库**：腾讯云 MySQL（用于持久化存储用户的生图任务和订单历史）。
    

#### 2. 前端路由结构 (基于 Nuxt 的自动路由机制)

- **`/` (Home 首页)**：
    
    - **功能**：产品品牌展示（类似 Mokika 的高级感界面），向用户展示不同的写真风格模板（如：赛博朋克、职场肖像、国风肖像）。
        
    - **交互**：用户点击某个风格模板后，自动跳转到生成页。
        
- **`/generate` (生成与结果页)**：
    
    - **功能**：上传原图、选择/切换风格、展示生图状态、对比原图与效果图、一键下载。
        

### 二、 核心业务流程：前端直传 COS ➔ 异步生图入库

为了防止大量用户同时上传几 MB 的照片导致你的服务器带宽瘫痪，项目采用**前端直接把图片推给腾讯云 COS** 的行业标准方案。

#### 1. 完整的业务时序流向

1. **申请通行证**：用户在前端点击上传图片，前端拦截文件流，先向 Nuxt 后端接口发起一个请求。后端通过腾讯云的 `STS（临时密钥服务）` 生成一个几分钟内有效的临时通行证。
    
2. **前端直传**：前端拿到通行证后，直接调用腾讯云前端 SDK，把照片推送到存储桶的 `inputs/`（原图）目录下。
    
3. **找回并渲染原图**：上传成功后，COS 会立刻返回这张图的公网 URL。前端拿到这个 URL 赋值给 Vue 变量，网页上就能**瞬间把这张原图找出来并展示**给用户看。
    
4. **后端拼接与调起生图**：用户点击“开始生图”，前端把**原图 URL** 和**所选风格**传给 Nuxt 后端。后端根据风格，在代码里找出**程序员内置的提示词优化公式**（例如：“高清、特定衣服、专业影棚灯光”），然后把内置提示词与原图 URL 一起作为参数，调用第三方的 AI 生图 API。
    
5. **拉回结果并入库**：第三方 AI 生成完毕后会返回一个图片链接。Nuxt 后端把这张效果图下载下来，转存到自己腾讯云 COS 的 `outputs/`（结果图）目录下。接着，后端连接**腾讯云 MySQL**，将本次生图的记录（原图链接、效果图链接、风格、时间）安全写入数据库。
    
6. **前端动态加字展示**：后端把最终属于你自己的 COS 效果图 URL 返回给前端。前端展示结果，并利用**腾讯云数据万象**功能，直接在图片 URL 后面拼接文字参数（不需要后端写画图代码），图片上就会自动实时渲染出指定的文字（如“Mokika AI 定制”）。
    

### 三、 腾讯云 MySQL 数据库设计

在腾讯云数据库控制台创建名为 `ai_photo_db` 的数据库，并建立以下表结构来记录每一笔生图业务：

#### 任务记录表：`photo_tasks`

|**字段名**|**数据类型**|**允许为空**|**默认值**|**含义/作用**|
|---|---|---|---|---|
|`id`|`INT`|否|自增 (Primary Key)|唯一任务 ID，用于索引|
|`style`|`VARCHAR(50)`|否|-|用户选定的写真风格（如 `cyberpunk`）|
|`user_image_url`|`VARCHAR(512)`|否|-|**从 COS 找出来的**用户原图网络链接|
|`ai_image_url`|`VARCHAR(512)`|是|`NULL`|AI 生成并转存到自家 COS 的效果图网络链接|
|`status`|`VARCHAR(20)`|否|`'processing'`|任务状态：`processing` (生成中), `success` (成功), `failed` (失败)|
|`created_at`|`TIMESTAMP`|否|`CURRENT_TIMESTAMP`|订单任务创建的精确时间|

### 四、 高级细节功能：图片加字与持久化

#### 1. 怎么完美实现“图片加上文字”？

传统的做法是在后端用 Python (PIL) 或 Node.js (Canvas) 去读写图片并往上写字，这非常消耗服务器的 CPU。

- **高效解法**：由于使用了腾讯云 COS，它自带了**数据万象 (CI)** 图像处理能力。
    
- **具体细节**：当你的 AI 效果图存放在 COS 的地址为 `https://xxx.cos.ap-shanghai.myqcloud.com/outputs/result.png` 时，前端在展示这张图的 `<img>` 标签里，只需要在链接后面直接拼接参数：
    
    Plaintext
    
    ```
    ?watermark/2/text/Base64加密后的文字/fontsize/30/fill/#ffffff
    ```
    
    腾讯云的 CDN 节点收到这个请求后，会自动在云端帮你在图片上叠加上指定的文字，并把加好字的图吐给前端。这不仅实现了“动态加字”，还完全不占你服务器的任何算力。
    

#### 2. 程序员内置提示词（Prompt）的设计策略

用户上传的照片千奇百怪。为了让生成效果像 Mokika 一样稳定，不至于“货不对板”，后端需要做 **Prompt 强制对齐**：

- 后端建立一个风格映射表。例如用户在前端选了“精致职场证件照”。
    
- 后端收到后，自动拿出程序员内置的保底提示词：`"Professional studio lighting, ID photo passport style, formal suit, centered, highly detailed, 8k resolution"`。
    
- 这样，无论用户原图的背景有多杂乱，AI 都会强制将其往“穿西装、纯色背景、专业摄影棚灯光”的风格上靠拢，保证生图的商用下限。
    

### 五、 项目的创建与落地过程

如果你现在要从零搭建这个系统，整个开发生命周期应该分为以下五个步骤：

#### 第一步：本地初始化基础设施

1. 使用前端脚手架在本地初始化一个标准的 **Nuxt 3** 空项目。
    
2. 通过 Nuxt 的模块配置文件（`nuxt.config`），将 **Element Plus** 组件库一键集成进去。
    
3. 在本地项目的根目录下创建一个 `.env` 配置文件，用来存放你腾讯云的 `SecretId`、`SecretKey`、`COS 存储桶名称/地域`、以及 `MySQL 的连接 IP、账号、密码`。
    

#### 第二步：设计后端 Nitro 路由 API

在 Nuxt 的 `server/api/` 目录下创建三个核心服务端接口文件（由于 Nuxt 的约定大于配置原则，创建文件即自动生成接口）：

1. **`cos-sts` 接口**：调用腾讯云的安全 SDK，负责为前端计算并分发临时上传凭证。
    
2. **`generate` 接口**：负责封装 MySQL 连接池。接收前端传来的原图 URL，处理内置 Prompt 拼接，执行第三方生图 API 请求，下载图片转存 COS，最终向 MySQL 的 `photo_tasks` 表里插入一条状态为 `success` 的数据。
    
3. **`history` 接口**：执行 SQL 语句 `SELECT * FROM photo_tasks ORDER BY created_at DESC`，用于让用户后续在个人中心查看自己历史做过的写真。
    

#### 第三步：构建前端交互页面

1. 在 `pages/` 目录下创建 `index.vue`（首页）和 `generate.vue`（核心生成页）。
    
2. 在生成页中，使用 Element Plus 的 `<el-select>` 让用户选择写真风格。
    
3. 使用 `<el-upload>` 组件，配置其拦截函数。在里面编写对接后端 `cos-sts` 获取临时密钥的逻辑，并用 COS 的 JS SDK 将文件推上云端。
    
4. **重点**：在直传成功的回调函数里，把得到的 COS URL 赋给前端的图片组件，实现“上传后立刻找回并渲染原图”的视觉体验。
    

#### 第四步：联调与本地测试

1. 在本地启动开发服务器，测试图片是否能成功直传到腾讯云 COS 桶中。
    
2. 尝试点击生图，检查后端是否成功把原图 URL 发送给了第三方生图 API。
    
3. 打开你的 MySQL 客户端（如 Navicat 或 DBeaver），观察每次生成完后，`photo_tasks` 表里有没有正确写入原图和 AI 效果图的链接。
    

#### 第五步：迁移与生产上线

当本地全流程跑通后，利用 Nuxt 的编译命令将项目打包。将其部署到你的腾讯云服务器（CVM）上，并在腾讯云控制台为你的域名配置好 **DNS（A 记录指向服务器）** 和 **CNAME 记录（指向 COS 边缘加速网络）**，开启防盗链，网站就可以正式对外商业化运营了。



# 提示词
### 一、 通用画质与反向提示词大底（全局封装）

在实际调用 AI 接口时，为了保证室内和建筑设计的高级渲染质感，后端应在每个功能提示词前自动拼上**通用正向大底**，并在每次请求中附带**反向大底**：

- **系统通用正向大底 (Global Positive Base)**：
    
    > `masterpiece, best quality, ultra-detailed, 8k resolution, photorealistic, ray-traced lighting, realistic shadow casting, complex global illumination, award-winning architectural photography,`
    
- **系统通用反向大底 (Global Negative Base)**：
    
    > `bad quality, worst quality, low resolution, blurry, noisy, grainy, distorted architectural structure, unrealistic reflection, text, watermark, signature, draft, canvas edges, logo, overexposed, pure black background, monochrome, low contrast, corrupted textures, amateur 3d render`
### 二、 全套 18 个核心功能卡片内置提示词矩阵

#### 📂 模块 1：室内效果图设计（Indoor Effects & Adjustments）

##### 1. 装饰灯 - 休闲氛围照明 (`indoor_deco_leisure`)

- **正向提示词 (Prompt)**：`cozy indoor lighting, elegant decorative pendant light emitting soft warm light, amber ambient glow, relaxed luxury residential atmosphere, cinematic side lighting, premium fabric and leather textures, interior design trend`
    
- **反向提示词 (Negative)**：`bright daylight, fluorescent light, overexposed, harsh direct light, flash photography, cold blue tone`
    

##### 2. 焦点灯光场景模式 (`indoor_spotlight_focus`)

- **正向提示词 (Prompt)**：`precise ceiling spotlights, focused beams illuminating key architectural areas, narrow beam angle, dramatic chiaroscuro contrast, high-end museum gallery lighting, visible light cones, sharp shadow definitions`
    
- **反向提示词 (Negative)**：`flat lighting, floodlight, shadowless, low contrast, uniformly illuminated room`
    

##### 3. 关闭天花灯带照明 (`indoor_strip_turn_off`)

- **正向提示词 (Prompt)**：`empty and unlit ceiling hidden cove, LED light strips completely turned off, minimal low-level linear lighting, subtle floor lamps and soft wall sconces glow, moody evening shadow atmosphere, clean architectural surfaces`
    
- **反向提示词 (Negative)**：`bright ceiling cove, shining linear lights, fully illuminated ceiling, morning sun`
    

##### 4. 灯光手绘图转实景灯光 (`indoor_sketch_to_real`)

- **正向提示词 (Prompt)**：`transform sketch into real luminous lighting, architectural LED linear profiles, modern recessed downlights, high-end V-ray interior rendering style, realistic glass and marble material reflections, tangible light sources`
    
- **反向提示词 (Negative)**：`visible lines, black and white sketch marks, hand-drawn background, unrefined surfaces, flat cartoon color`
    

##### 5. 一键智能灯光场景设计（给参考图） (`indoor_smart_scene`)

- **正向提示词 (Prompt)**：`complete high-end intelligent lighting design, luxury hotel lobby ambiance, balanced direct and indirect lighting layout, multi-layered color temperatures, realistic light propagation on walls, professional soft diffusion`
    
- **反向提示词 (Negative)**：`single light source, unlit corners, chaotic shadows, neon colors, rave lighting`
    

##### 6. 提示词精准设计室内灯光 (`indoor_prompt_precision`)

- **正向提示词 (Prompt)**：`custom high-fidelity lighting arrangement, sophisticated light fixture placement, detailed geometric shadows, soft light scattering, architectural digest magazine style presentation, high dynamic range`
    
- **反向提示词 (Negative)**：`random light leaks, volumetric fog overdone, low dynamic range, dark spots`
    

##### 7. 根据提示词准确得出灯光效果 (`indoor_prompt_matching`)

- **正向提示词 (Prompt)**：`accurate light spectrum simulation, realistic physics-based rendering, clear illumination boundaries, distinct contrast between illuminated areas and shadows, crisp and clean architectural visualization`
    
- **反向提示词 (Negative)**：`vague light sources, floating artifacts, light bleeding through solid objects`
    

##### 8. 写提示词更改局部灯光色温 (`indoor_color_temp_tweak`)

- **正向提示词 (Prompt)**：`architectural indoor color temperature adjustment, white balance fine-tuning, realistic warm and cold light interaction, smooth color bleeding on matte walls, exact Kelvin temperature look`
    
- **反向提示词 (Negative)**：`oversaturated neon, artificial rainbow colors, extreme tint, black and white`
    

##### 9. 一键调光功能调整灯光亮度 (`indoor_dimmer_control`)

- **正向提示词 (Prompt)**：`controlled lumen adjustment, relative brightness scaling, accurate dimming simulation, soft shadow transitions, preserved texture details in shadow areas, ambient dark luxury`
    
- **反向提示词 (Negative)**：`pitch black, overexposed white blocks, washed out colors`
    

##### 10. 提亮整个室内空间灯光亮度 (`indoor_brighten_all`)

- **正向提示词 (Prompt)**：`fully illuminated spatial volume, bright ambient fill light, shadowless corner enhancement, high-key lighting design, commercial interior presentation, clean and energetic atmosphere`
    
- **反向提示词 (Negative)**：`dim lighting, heavy vignette, gloomy atmosphere, dark corners, moody shadows`
    

##### 11. 一键去除天花原有灯具 (`indoor_erase_fixtures` - _重绘消除_)

- **正向提示词 (Prompt)**：`clean and smooth gypsum ceiling surface, completely empty ceiling architecture, seamless drywall finish, no lamps, no tracks, original structural shadow lines preserved`
    
- **反向提示词 (Negative)**：`remaining wires, holes, broken glass, ghost artifacts, light spots where lamp was`
    

#### 📂 模块 2：精准位置控制（Canvas 涂抹/标线加灯）

##### 12. 精准位置/室内任意加灯带照明 (`control_strip_light` - _红线约束_)

- **正向提示词 (Prompt)**：`a continuous seamless LED light strip installed precisely along the marked line, hidden cove illumination, elegant grazing light effect flowing down the wall, 4000k linear light profile, crisp architectural linear alignment`
    
- **反向提示词 (Negative)**：`broken light lines, dots, crooked light strips, bulbs showing`
    

##### 13. 任意加射灯照明/指定位置加灯具 (`control_spotlight_lamp` - _红点/圈选约束_)

- **正向提示词 (Prompt)**：`a sleek modern minimalist spotlight installed exactly on the marked position, down-pointing conical light beam, sharp accent illumination, metallic fixture casing seamlessly mounted to the ceiling`
    
- **反向提示词 (Negative)**：`floating lamps, disconnected wires, distorted lamp shape, multiple overlapping fixtures`
    

##### 14. 室内任意加磁吸灯/特定产品 (`control_magnetic_track`)

- **正向提示词 (Prompt)**：`a high-end black magnetic track lighting rail integrated into the ceiling, slim linear track light modules inserted, continuous warm glow, high-end studio lighting hardware style`
    
- **反向提示词 (Negative)**：`clunky traditional chandeliers, retro styles, loose parts, messy ceiling`
    

#### 📂 模块 3：建筑效果图与户外亮化（Outdoor & Facade)

##### 15. 真实感白天转夜景通用 (`outdoor_day_to_night`)

- **正向提示词 (Prompt)**：`dusk evening architectural photography, deep blue hour sky background with subtle stars, beautiful warm golden lights glowing from inside building windows, interior illumination bleeding out, realistic glass facade reflections, high-end night appearance`
    
- **反向提示词 (Negative)**：`bright daylight, sunshine, blue sky with white clouds, sharp noon shadows, sun rays`
    

##### 16. 白天转夜景只出环境光 (`outdoor_night_ambient_only`)

- **正向提示词 (Prompt)**：`night architectural silhouette, dim ambient environment lighting, twilight horizon sky background, subtle accent lighting on key columns, low contrast external light, emphasis on architectural form and geometry shadow`
    
- **反向提示词 (Negative)**：`bright floodlights, multi-colored flashing lights, fully illuminated facade, harsh neon signs`
    

##### 17. 现代建筑幕墙灯光设计 (`outdoor_curtain_wall`)

- **正向提示词 (Prompt)**：`modern glass skyscraper facade lighting, linear LED vertical profiles running upwards, grazing wall wash light effect on concrete columns, tech-style architectural illumination, sharp clean geometric light beams, commercial building mood`
    
- **反向提示词 (Negative)**：`classic European building, messy chaotic wires, round vintage lantern lamps, flickering lights`
    

##### 18. 景观建筑夜景照明设计/风格迁移 (`outdoor_landscape_style`)

- **正向提示词 (Prompt)**：`premium landscape illumination design, reference-grade light distribution on structural facades, elegant luxury property night presentation, harmonious interplay of warm white and neutral white light sources`
    
- **反向提示词 (Negative)**：`cheap street lights, security floodlights, green or red laser lights`
    

##### 19. 只给树木加投光灯 (`outdoor_tree_uplighting`)

- **正向提示词 (Prompt)**：`landscape greening uplighting, ground-buried waterproof spotlights projecting upwards, illuminating botanical tree leaves and complex branches, vibrant organic textures, crisp light beams passing through green foliage`
    
- **反向提示词 (Negative)**：`ambient daylight, uniformly lit sky, unlit trees, floating lanterns`
    

##### 20. 指定挂饰灯生成树木亮化 (`outdoor_tree_hanging_deco`)

- **正向提示词 (Prompt)**：`romantic landscape lighting, sparkling fairy light strings wrapped around tree branches, micro-glowing light particles, starry golden holiday decoration look, beautiful bokeh effects in the dark background`
    
- **反向提示词 (Negative)**：`industrial construction site floodlights, office lighting, minimal cold style`
    

#### 📂 模块 4：线稿互转（Structure Conversion）

##### 21. 线稿图/草图转 3D 效果图 (`sketch_to_3d` - _ControlNet 约束_)

- **正向提示词 (Prompt)**：`complete architectural visualization, filling materials with high-end realistic textures, realistic wood paneling and polished marble floors, balanced luxury interior layer lighting, photorealistic 3d render presentation`
    
- **反向提示词 (Negative)**：`remaining lines, pencil draft marks, empty untextured blocks, sketch background`
    

##### 22. 效果图转线稿手绘底图 (`image_to_sketch` - _大模型兜底模板_)

- **正向提示词 (Prompt)**：`pure plain white background, clean crisp black and white lineart, sharp architectural structural contour lines, zero colors, zero shadows, perfect blueprint wireframe template for interior designers`
    
- **反向提示词 (Negative)**：`color gradients, grayscale shading, 3d volumetric render, photographic textures`