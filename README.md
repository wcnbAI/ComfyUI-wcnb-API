# ComfyUI-wcnb-API

中文 | [English](README_EN.md)

wcnb基于nanobanana制作ComfyUI文生图和图生文插件。

## 安装说明

### 方法一：手动安装

1. 将此存储库克隆到ComfyUI的`custom_nodes`目录：
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/WcnbAi/ComfyUI-wcnb-API
   ```

2. 安装所需依赖：

   如果你使用ComfyUI便携版
   ```
   ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```

   如果你使用自己的Python环境
   ```
   path\to\your\python.exe -m pip install -r requirements.txt
   ```

### 方法二：通过ComfyUI Manager安装

1. 在ComfyUI中安装并打开ComfyUI Manager
2. 在Manager中搜索"wcnb API"
3. 点击安装按钮

安装完成后重启 ComfyUI

## 节点说明

### wcnb image

示例工作流文件位于 `workflow` 目录，可直接在 ComfyUI 中加载体验。

通过 wcnb API 生成图像的节点。

**输入参数：**
- **prompt** (必填)：描述你想要生成的图像的文本提示词
- **api_key** (必填)：你的 wcnb API 密钥（首次设置后会自动保存）
- **model**：模型选择
- **aspect_ratio**：选择图像方向（自由比例、横屏、竖屏、方形）
- **temperature**：控制生成多样性的参数（0.0-2.0）
- **seed** (可选)：随机种子，指定值可重现结果
- **images** (可选)：参考图像输入，支持多张图片

**输出：**
- **image**：生成的图像，可以连接到ComfyUI的其他节点
- **API Respond**：包含处理日志和API返回的文本信息

**使用场景：**
- 创建独特的概念艺术
- 基于文本描述生成图像
- 使用一张或多张参考图像创建风格一致的新图像
- 基于图像的编辑操作

**多图片功能说明：**
- 节点现在支持同时输入多张参考图像
- 多张图像将一起发送给 wcnb API 作为风格参考
- 系统会自动调整提示词，告知模型有多张参考图像
- 此功能非常适合混合多种风格或提供更多参考信息

## 获取API密钥

1. 访问[wcnb.ai](https://wcnb.ai)
2. 创建一个账户或登录
3. 在控制台的「令牌管理」中获取API密钥
4. 复制API密钥并粘贴到 ComfyUI 插件节点的 `api_key` 参数中（只需首次输入，之后会自动保存）

## 联系客服

<img src="assets/wechat_qr.png" alt="微信扫码咨询客服" width="50%" />

## 温度参数说明

- 温度值范围：0.0到2.0
- 较低的温度（接近0）：生成更确定性、可预测的结果
- 较高的温度（接近2）：生成更多样化、创造性的结果
- 默认值1.0：平衡确定性和创造性

## 注意事项

- API 可能有使用限制或费用，请查阅 wcnb 的官方文档
- 图像生成质量和速度取决于 wcnb 的服务器状态和您的网络连接
- 参考图像功能会将您的图像提供给 wcnb 服务，请注意隐私影响
- 首次使用时需要输入 API 密钥，之后会自动存储在节点目录中的 `wcnb_api_key.txt` 文件中
- 关于图像方向，wcnb API 会根据选择的方向（横屏、竖屏或方形）生成适合的图像（但模型不一定完全按要求生成）

