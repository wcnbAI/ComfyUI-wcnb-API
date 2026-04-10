import time
import tempfile
import re
import os
import base64
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import List, Union, Optional, Any
import traceback
import subprocess
import json
import urllib.request
import urllib.parse

# 尝试导入 ComfyUI 工具模块
try:
    import comfy.utils
    HAS_COMFY_UTILS = True
except ImportError:
    HAS_COMFY_UTILS = False
    # 创建一个简单的进度条模拟类
    class SimpleProgressBar:
        def __init__(self, total):
            self.total = total
            self.current = 0
        
        def update_absolute(self, value):
            self.current = value
            print(f"进度: {value}%")

# 尝试导入 ComfyUI 视频相关模块
try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

try:
    from comfy_api.input_impl import VideoFromFile
    HAS_VIDEO_FROM_FILE = True
except ImportError:
    HAS_VIDEO_FROM_FILE = False

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Enter your Google AI API key"}),

                "base_url": (["https://wcnb.ai"], {"default": "https://wcnb.ai"}),
                "model_name": ([
                    "gemini-3-pro-image-preview",
                    "自定义输入 (Custom Input)"
                ], {"default": "gemini-3-pro-image-preview"}),
                "custom_model_name": ("STRING", {"default": "", "multiline": False, "placeholder": "输入自定义模型名称"}),
                "imageSize": ([
                    "1K",
                    "2K",
                    "4K"
                ], {"default": "1K"}),
                "aspect_ratio": ([
                    "Free (自由比例)",
                    "Landscape (横屏)",
                    "Portrait (竖屏)",
                    "Square (方形)",
                ], {"default": "Free (自由比例)"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "wcnb"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        self.url_file = os.path.join(self.node_dir, "gemini_base_url.txt")
        self.model_file = os.path.join(self.node_dir, "gemini_model_name.txt")
    
    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥"""
        # 如果用户输入了有效的密钥，使用并保存
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 保存到文件中
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("已保存API密钥到节点目录")
            except Exception as e:
                self.log(f"保存API密钥失败: {e}")
            return user_input_key
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("使用已保存的API密钥")
                    return saved_key
            except Exception as e:
                self.log(f"读取保存的API密钥失败: {e}")
                
        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥")
        return ""

    def tensor2pil(self,image: torch.Tensor) -> List[Image.Image]:
        """
        Convert tensor to PIL image(s), matching ComfyUI's implementation.

        Args:
            image: Tensor with shape [B, H, W, 3] or [H, W, 3], values in range [0, 1]

        Returns:
            List[Image.Image]: List of PIL Images
        """
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        if batch_count > 1:
            out = []
            for i in range(batch_count):
                out.extend(self.tensor2pil(image[i]))
            return out

        # Convert tensor to numpy array, scale to [0, 255], and clip values
        numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        return [Image.fromarray(numpy_image)]

    def pil2tensor(self,image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Convert PIL image(s) to tensor, matching ComfyUI's implementation.
        Optimized for large images.

        Args:
            image: Single PIL Image or list of PIL Images

        Returns:
            torch.Tensor: Image tensor with values normalized to [0, 1]
        """
        if isinstance(image, list):
            if len(image) == 0:
                return torch.empty(0)
            return torch.cat([self.pil2tensor(img) for img in image], dim=0)

        # Convert PIL image to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Optimized conversion: use asarray to avoid unnecessary copy when possible
        # For large images, this is still necessary but we can optimize the process
        img_array = np.asarray(image, dtype=np.float32) / 255.0

        # Return tensor with shape [1, H, W, 3]
        # Use contiguous memory for better performance
        return torch.from_numpy(img_array).contiguous()[None,]
    
    def base64_to_tensor(self, base64_data: str) -> torch.Tensor:
        """
        直接从 Base64 字符串转换为 ComfyUI 兼容的 Tensor，优化性能。
        使用 PIL 的 tobytes() 直接获取字节数据，然后通过 NumPy 快速转换为 tensor。
        这比原来的方法更快，因为减少了不必要的内存复制。
        
        Args:
            base64_data: Base64 编码的图像字符串（可能包含 data URI 前缀）
            
        Returns:
            torch.Tensor: 图像 tensor，形状为 [1, H, W, 3]，值范围 [0, 1]
        """
        import time
        start_time = time.time()
        
        # 移除可能的 data URI 前缀
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Base64 解码
        step_start = time.time()
        image_bytes = base64.b64decode(base64_data)
        self.log(f"[性能] Base64 解码耗时: {time.time() - step_start:.2f} 秒，数据大小: {len(image_bytes) / 1024:.2f} KB")
        
        # 使用 PIL 解码图像格式（这是必需的，因为需要解码 JPEG/PNG 等格式）
        step_start = time.time()
        pil_image = Image.open(BytesIO(image_bytes))
        self.log(f"[性能] PIL 图像打开耗时: {time.time() - step_start:.2f} 秒，尺寸: {pil_image.size}, 模式: {pil_image.mode}")
        
        # 确保是 RGB 模式
        if pil_image.mode != 'RGB':
            step_start = time.time()
            pil_image = pil_image.convert('RGB')
            self.log(f"[性能] 图像模式转换耗时: {time.time() - step_start:.2f} 秒")
        
        # 获取图像尺寸
        width, height = pil_image.size
        
        # 优化方法：使用 PIL 的 tobytes() 直接获取字节数据
        # 然后直接从字节创建 NumPy 数组，避免 PIL → NumPy 的额外复制
        try:
            self.log(f"[解码方法] 使用优化的 Base64 → Tensor 转换（tobytes + frombuffer）")
            
            # 获取原始字节数据（uint8 格式，RGB 顺序，H*W*3 字节）
            step_start = time.time()
            img_bytes = pil_image.tobytes()
            self.log(f"[性能] PIL.tobytes() 耗时: {time.time() - step_start:.2f} 秒，字节数: {len(img_bytes) / 1024:.2f} KB")
            
            # 直接从字节创建 NumPy 数组（uint8），然后重塑为 [H, W, 3]
            # 这比 np.asarray(pil_image) 更快，因为避免了 PIL 内部的转换
            step_start = time.time()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape(height, width, 3)
            self.log(f"[性能] np.frombuffer() 耗时: {time.time() - step_start:.2f} 秒")
            
            # 转换为 float32 并归一化（在 NumPy 中完成，比在 PyTorch 中更快）
            step_start = time.time()
            img_array = img_array.astype(np.float32) / 255.0
            self.log(f"[性能] 类型转换和归一化耗时: {time.time() - step_start:.2f} 秒")
            
            # 转换为 PyTorch tensor
            step_start = time.time()
            img_tensor = torch.from_numpy(img_array).contiguous()
            self.log(f"[性能] torch.from_numpy() 耗时: {time.time() - step_start:.2f} 秒")
            
            # 返回形状为 [1, H, W, 3] 的 tensor
            total_time = time.time() - start_time
            self.log(f"[解码方法] 优化转换成功，图像尺寸: {img_tensor.shape}")
            self.log(f"[性能] 总耗时: {total_time:.2f} 秒")
            return img_tensor.unsqueeze(0)
        except Exception as e:
            # 如果优化方法失败，回退到原来的方法
            self.log(f"[解码方法] 优化转换失败，回退到标准方法（asarray）: {str(e)}")
            return self.pil2tensor(pil_image)

    def get_base_url(self, user_input_url):
        """获取Base URL，优先使用用户输入的URL"""
        # 如果用户输入了有效的URL，使用并保存
        if user_input_url and user_input_url.strip():
            url = user_input_url.strip()
            self.log(f"使用用户输入的Base URL: {url}")
            # 保存到文件中
            try:
                with open(self.url_file, "w") as f:
                    f.write(url)
                self.log("已保存Base URL到节点目录")
            except Exception as e:
                self.log(f"保存Base URL失败: {e}")
            return url
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.url_file):
            try:
                with open(self.url_file, "r") as f:
                    saved_url = f.read().strip()
                if saved_url:
                    self.log(f"使用已保存的Base URL: {saved_url}")
                    return saved_url
            except Exception as e:
                self.log(f"读取保存的Base URL失败: {e}")
                
        # 默认URL
        default_url = "https://wcnb.ai"
        self.log(f"使用默认Base URL: {default_url}")
        return default_url
    
    def get_model_name(self, user_input_model):
        """获取模型名称，优先使用用户输入的模型"""
        # 如果用户输入了有效的模型名，使用并保存
        if user_input_model and user_input_model.strip():
            model = user_input_model.strip()
            self.log(f"使用用户输入的模型: {model}")
            # 保存到文件中
            try:
                with open(self.model_file, "w") as f:
                    f.write(model)
                self.log("已保存模型名称到节点目录")
            except Exception as e:
                self.log(f"保存模型名称失败: {e}")
            return model
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, "r") as f:
                    saved_model = f.read().strip()
                if saved_model:
                    self.log(f"使用已保存的模型: {saved_model}")
                    return saved_model
            except Exception as e:
                self.log(f"读取保存的模型名称失败: {e}")
                
        # 默认模型
        default_model = "models/gemini-2.0-flash-exp"
        self.log(f"使用默认模型: {default_model}")
        return default_model
    
    def remove_base64_prefix(self, base64_string):
        """移除 base64 字符串的 data URI 前缀"""
        if base64_string.startswith('data:'):
            # 移除 data:image/jpeg;base64, 或类似的前缀
            return base64_string.split(',', 1)[1] if ',' in base64_string else base64_string
        return base64_string

    def extract_image_urls(self, response_text):
        """从响应文本中提取图像URL，支持多种格式"""
        if not response_text:
            return []
        
        # 方式1: Markdown格式的图片链接 ![...](url)
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)
        
        if not matches:
            # 方式2: 直接URL格式
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
        
        return matches if matches else []

    def image_url_to_base64(self,image_url):
        """将图片URL指向的资源编码为Base64"""
        if image_url is None:
            return None
        try:
            # 发送HTTP GET请求获取图片二进制数据
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()  # 检查HTTP状态码
            # 执行Base64编码
            base64_str = base64.b64encode(response.content).decode('utf-8')
            # 添加Base64图片前缀（可直接在HTML中使用）
            return f"data:image/{response.headers['Content-Type'].split('/')[1]};base64,{base64_str}"
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return None
    def extract_image_base64(self, response_text):
        """从响应文本中提取 base64 图像数据，支持多种格式"""
        if response_text is None:
            self.log("响应文本为None，无法提取base64字符串")
            return None
        
        # 方式1: 提取完整的 data URI，然后手动分离 base64 部分
        # 匹配 data:image/xxx;base64, 后面的所有内容，直到遇到非base64字符或文本结束
        data_uri_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)'
        matches = re.findall(data_uri_pattern, response_text)
        if matches:
            # 选择最长的匹配（如果有多个）
            base64_str = max(matches, key=len)
            # 清理空白字符
            base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
            # 移除末尾可能的不完整填充
            base64_str = base64_str.rstrip('=')
            # 重新添加必要的填充（base64 必须是4的倍数）
            padding = (4 - len(base64_str) % 4) % 4
            base64_str += '=' * padding
            
            self.log(f"从 data URI 格式中提取到 base64 数据，长度: {len(base64_str)}")
            # 验证 base64 是否有效
            try:
                decoded = base64.b64decode(base64_str)
                self.log(f"Base64 解码成功，解码后大小: {len(decoded)} 字节")
                return base64_str
            except Exception as e:
                self.log(f"提取的 base64 数据解码失败: {str(e)}")
                # 如果第一个匹配失败，尝试其他匹配
                for match in matches:
                    if match != matches[0]:
                        base64_str = match.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
                        base64_str = base64_str.rstrip('=')
                        padding = (4 - len(base64_str) % 4) % 4
                        base64_str += '=' * padding
                        try:
                            decoded = base64.b64decode(base64_str)
                            self.log(f"使用备用匹配成功，base64 数据长度: {len(base64_str)}，解码后大小: {len(decoded)} 字节")
                            return base64_str
                        except:
                            continue
        
        # 方式2: Markdown 格式中的 data URI ![...](data:image/xxx;base64,xxxxx)
        markdown_pattern = r'!\[.*?\]\((data:image/[^)]+)\)'
        markdown_matches = re.findall(markdown_pattern, response_text)
        if markdown_matches:
            for data_uri in markdown_matches:
                # 从 data URI 中提取 base64
                if 'base64,' in data_uri:
                    base64_str = data_uri.split('base64,', 1)[1]
                    # 清理空白字符
                    base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
                    # 移除末尾可能的不完整填充
                    base64_str = base64_str.rstrip('=')
                    # 重新添加必要的填充
                    padding = (4 - len(base64_str) % 4) % 4
                    base64_str += '=' * padding
                    try:
                        decoded = base64.b64decode(base64_str)
                        self.log(f"从 Markdown data URI 中提取到 base64 数据，长度: {len(base64_str)}，解码后大小: {len(decoded)} 字节")
                        return base64_str
                    except Exception as e:
                        self.log(f"Markdown data URI 中的 base64 解码失败: {str(e)}")
                        continue
        
        # 方式3: 纯 base64 字符串（可能包含换行和空格）
        base64_pattern = r'([A-Za-z0-9+/]{100,}={0,2})'  # 至少100个字符的base64字符串
        matches = re.findall(base64_pattern, response_text)
        if matches:
            # 选择最长的匹配（通常是图像数据）
            base64_str = max(matches, key=len).strip().replace('\n', '').replace(' ', '').replace('\r', '')
            # 验证是否是有效的 base64
            try:
                decoded = base64.b64decode(base64_str)
                self.log(f"从纯 base64 格式中提取到数据，长度: {len(base64_str)}，解码后大小: {len(decoded)} 字节")
                return base64_str
            except:
                pass
        
        self.log("未找到base64字符串")
        return None
    def resize_to_target_size(self, image, target_size):
        """Resize image to target size while preserving aspect ratio with padding"""
        img_width, img_height = image.size
        target_width, target_height = target_size
        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        scale = min(width_ratio, height_ratio)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y))
        return new_img
    def save_base64_to_file(self, base64_data, prefix="gemini_output"):
        """将 base64 数据保存为本地文件并返回路径"""
        try:
            # 移除可能的 data URI 前缀
            clean_base64 = self.remove_base64_prefix(base64_data)
            
            # 解码 base64 数据
            image_bytes = base64.b64decode(clean_base64)
            
            # 创建临时文件
            timestamp = int(time.time())
            filename = f"{prefix}_{timestamp}.png"
            
            # 获取 ComfyUI 的输出目录
            output_dir = os.path.join(os.path.dirname(os.path.dirname(self.node_dir)), "output")
            if not os.path.exists(output_dir):
                output_dir = tempfile.gettempdir()
            
            filepath = os.path.join(output_dir, filename)
            
            # 保存图像
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            self.log(f"图像已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            self.log(f"保存图像失败: {e}")
            return None
    
    def load_image_from_file(self, filepath):
        """从文件路径加载图像并转换为 ComfyUI 格式"""
        try:
            pil_image = Image.open(filepath)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为 ComfyUI 格式
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            self.log(f"成功从文件加载图像: {img_tensor.shape}")
            return img_tensor
            
        except Exception as e:
            self.log(f"从文件加载图像失败: {e}")
            return None
    
    def generate_empty_image(self, width=512, height=512):
        """生成标准格式的空白RGB图像张量 - 使用默认尺寸"""
        # 根据比例设置默认尺寸
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"创建ComfyUI兼容的空白图像: 形状={tensor.shape}, 类型={tensor.dtype}")
        return tensor
    
    def validate_and_fix_tensor(self, tensor, name="图像"):
        """验证并修复张量格式，确保完全兼容ComfyUI"""
        try:
            # 基本形状检查
            if tensor is None:
                self.log(f"警告: {name} 是None")
                return None
                
            self.log(f"验证 {name}: 形状={tensor.shape}, 类型={tensor.dtype}, 设备={tensor.device}")
            
            # 确保形状正确: [B, C, H, W]
            if len(tensor.shape) != 4:
                self.log(f"错误: {name} 形状不正确: {tensor.shape}")
                return None
                
            if tensor.shape[1] != 3:
                self.log(f"错误: {name} 通道数不是3: {tensor.shape[1]}")
                return None
                
            # 确保类型为float32
            if tensor.dtype != torch.float32:
                self.log(f"修正 {name} 类型: {tensor.dtype} -> torch.float32")
                tensor = tensor.to(dtype=torch.float32)
                
            # 确保内存连续
            if not tensor.is_contiguous():
                self.log(f"修正 {name} 内存布局: 使其连续")
                tensor = tensor.contiguous()
                
            # 确保值范围在0-1之间
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if min_val < 0 or max_val > 1:
                self.log(f"修正 {name} 值范围: [{min_val}, {max_val}] -> [0, 1]")
                tensor = torch.clamp(tensor, 0.0, 1.0)
                
            return tensor
        except Exception as e:
            self.log(f"验证张量时出错: {e}")
            traceback.print_exc()
            return None
    
    def image_to_base64(self, image):
        """将PIL图像转换为base64字符串"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_image(self, prompt, api_key, base_url, model_name, custom_model_name, imageSize, aspect_ratio, temperature, seed=-1,
                       image1=None,image2=None,image3=None,image4=None,image5=None,image6=None,image7=None,image8=None,image9=None,image10=None,image11=None,image12=None):
        """生成图像 - 优先使用原生Gemini格式，失败后回退到OpenAI兼容格式"""
        import time
        function_start_time = time.time()
        response_text = ""
        
        # 重置日志消息
        self.log_messages = []
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            
            # 处理Base URL
            actual_base_url = self.get_base_url(base_url)
            
            # 处理模型名称：如果选择了"自定义输入"，使用 custom_model_name；否则使用 gemini-3-pro-image-preview
            if model_name == "自定义输入 (Custom Input)":
                actual_model_name = custom_model_name.strip() if custom_model_name and custom_model_name.strip() else "gemini-3-pro-image-preview"
                if not actual_model_name or actual_model_name == "":
                    actual_model_name = "gemini-3-pro-image-preview"
                    self.log("警告: 自定义模型名称为空，使用默认值 gemini-3-pro-image-preview")
                self.log(f"使用自定义模型: {actual_model_name}")
            else:
                actual_model_name = "gemini-3-pro-image-preview"
                self.log(f"使用模型: {actual_model_name}")
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message + "\n\n## 使用说明\n1. 在节点中输入您的Google API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                return (self.generate_empty_image(512, 512), full_text)
            
            # 处理种子值
            if seed == -1:
                import random
                seed = random.randint(0, 2147483647)
                self.log(f"生成随机种子值: {seed}")
            else:
                self.log(f"使用指定的种子值: {seed}")
            
            # 构建提示词，根据图像尺寸和宽高比
            if "Free" in aspect_ratio:
                simple_prompt = f"Create a detailed image of: {prompt}."
            elif "Landscape" in aspect_ratio:
                simple_prompt = f"Generate a wide rectangular image where width is greater than height. Create a detailed image of: {prompt}."
            elif "Portrait" in aspect_ratio:
                simple_prompt = f"Generate a tall rectangular image where height is greater than width. Create a detailed image of: {prompt}."
            else:  # Square
                simple_prompt = f"Generate a square image where width equals height. Create a detailed image of: {prompt}."
            
            self.log(f"使用温度值: {temperature}，种子值: {seed}")
            
            # 处理参考图像
            all_images = [img for img in [image1, image2, image3, image4,
                                          image5, image6, image7, image8,
                                          image9, image10, image11, image12]
                          if img is not None]
            reference_images_count = len(all_images)
            
            # 准备图像数据（两种格式都需要）
            image_base64_list = []
            if reference_images_count > 0:
                self.log(f"检测到 {reference_images_count} 张参考图像")
                try:
                    for i, img_tensor in enumerate(all_images):
                        pil_image = self.tensor2pil(img_tensor)[0]
                        image_base64 = self.image_to_base64(pil_image)
                        image_base64_list.append(image_base64)
                        self.log(f"成功处理参考图像 {i+1}")
                    
                    # 如果有参考图像，更新提示词
                    if reference_images_count == 1:
                        simple_prompt += " Use this reference image as guidance."
                    else:
                        simple_prompt += f" Use these {reference_images_count} reference images as guidance."
                except Exception as img_error:
                    self.log(f"参考图像处理错误: {str(img_error)}")
                    image_base64_list = []
            
            # ========== 尝试1: 原生 Gemini 格式 ==========
            self.log("=" * 50)
            self.log("尝试1: 使用原生 Gemini 格式")
            self.log("=" * 50)
            
            try:
                # 构建原生 Gemini 格式的请求
                parts = [{"text": simple_prompt}]
                
                # 添加参考图像（使用 inline_data 格式）
                for image_base64 in image_base64_list:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64  # 纯 base64，不带 data URI 前缀
                        }
                    })
                
                contents = [{
                    "role": "user",
                    "parts": parts
                }]
                
                # 构建 generationConfig
                generation_config = {
                            "temperature": temperature,
                            "seed": seed,
                            "responseModalities": ["TEXT", "IMAGE"]
                        }
                
                # 添加 imageConfig
                image_config = {}
                if imageSize:
                    image_config["imageSize"] = imageSize
                
                # 处理宽高比
                if "Landscape" in aspect_ratio:
                    image_config["aspectRatio"] = "16:9"
                elif "Portrait" in aspect_ratio:
                    image_config["aspectRatio"] = "9:16"
                elif "Square" in aspect_ratio:
                    image_config["aspectRatio"] = "1:1"
                
                if image_config:
                    generation_config["imageConfig"] = image_config
                
                # 原生 Gemini 格式的端点
                gemini_url = f"{actual_base_url.rstrip('/')}/v1beta/models/{actual_model_name}:generateContent"
                self.log(f"原生 Gemini 格式 - URL: {gemini_url}")
                self.log(f"原生 Gemini 格式 - 模型: {actual_model_name}, 图像尺寸: {imageSize}")
                
                gemini_payload = {
                    "contents": contents,
                    "generationConfig": generation_config
                }
                
                # 发送原生 Gemini 格式请求
                import time
                api_request_start = time.time()
                self.log(f"[性能] 开始发送 API 请求...")
                gemini_response = requests.post(
                    gemini_url,
                    headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": actual_api_key
                },
                    json=gemini_payload,
                        timeout=900
                    )
                api_request_time = time.time() - api_request_start
                self.log(f"[性能] API 请求耗时: {api_request_time:.2f} 秒 ({api_request_time/60:.2f} 分钟)")
                self.log(f"原生 Gemini 格式 - HTTP响应状态码: {gemini_response.status_code}")
                
                if gemini_response.status_code == 200:
                    gemini_result = gemini_response.json()
                    self.log("✅ 原生 Gemini 格式请求成功")
                    
                    # 添加详细的调试日志
                    try:
                        import json
                        response_str = json.dumps(gemini_result, ensure_ascii=False, indent=2)
                        # self.log(f"[调试] 完整响应内容 (前2000字符):\n{response_str[:2000]}")
                        self.log(f"[调试] 响应顶层字段: {list(gemini_result.keys())}")
                    except Exception as debug_error:
                        self.log(f"[调试] 打印响应内容失败: {str(debug_error)}")
                    
                    # 处理原生 Gemini 格式的响应
                    if gemini_result.get('candidates') and len(gemini_result['candidates']) > 0:
                        self.log(f"[调试] candidates 数量: {len(gemini_result['candidates'])}")
                        candidate = gemini_result['candidates'][0]
                        self.log(f"[调试] candidate 字段: {list(candidate.keys())}")
                        
                        if candidate.get('content'):
                            self.log(f"[调试] content 字段: {list(candidate['content'].keys())}")
                            if candidate['content'].get('parts'):
                                self.log(f"[调试] parts 数量: {len(candidate['content']['parts'])}")
                                for i, part in enumerate(candidate['content']['parts']):
                                    self.log(f"[调试] part[{i}] 字段: {list(part.keys())}")
                                    if 'text' in part:
                                        text_preview = str(part['text'])[:200] if part['text'] else "None"
                                        self.log(f"[调试] part[{i}].text 内容预览: {text_preview}")
                                    
                                    # 支持两种命名格式
                                    inline_data = None
                                    inline_data_key = None
                                    if 'inline_data' in part:
                                        inline_data = part['inline_data']
                                        inline_data_key = 'inline_data'
                                        self.log(f"[调试] part[{i}] 使用 inline_data 格式")
                                    elif 'inlineData' in part:
                                        inline_data = part['inlineData']
                                        inline_data_key = 'inlineData'
                                        self.log(f"[调试] part[{i}] 使用 inlineData 格式")
                                    
                                    if inline_data:
                                        self.log(f"[调试] part[{i}].{inline_data_key} 字段: {list(inline_data.keys())}")
                                        # 支持两种命名格式获取 mime_type（优化：先检查一个，找不到再检查另一个）
                                        mime_type = inline_data.get('mime_type') or inline_data.get('mimeType') or 'N/A'
                                        self.log(f"[调试] part[{i}].mime_type/mimeType: {mime_type}")
                                        data_value = inline_data.get('data')
                                        data_preview = str(data_value)[:100] if data_value else "None"
                                        self.log(f"[调试] part[{i}].data 预览: {data_preview}...")
                                    elif inline_data_key:
                                        self.log(f"[调试] part[{i}].{inline_data_key} 为 None")
                            else:
                                self.log(f"[调试] content.parts 不存在或为空")
                        else:
                            self.log(f"[调试] candidate.content 不存在")
                    else:
                        self.log(f"[调试] candidates 不存在或为空")
                        if 'candidates' in gemini_result:
                            self.log(f"[调试] candidates 值: {gemini_result['candidates']}")
                    
                    # 处理原生 Gemini 格式的响应（继续原有逻辑）
                    if gemini_result.get('candidates') and len(gemini_result['candidates']) > 0:
                        candidate = gemini_result['candidates'][0]
                        if candidate.get('content') and candidate['content'].get('parts'):
                            images = []
                            
                            for part_idx, part in enumerate(candidate['content']['parts']):
                                self.log(f"[调试] 处理 part[{part_idx}]: {list(part.keys())}")
                                
                                # 提取文本
                                if 'text' in part and part['text'] is not None:
                                    response_text += part['text']
                                    self.log(f"API返回文本: {part['text'][:100]}..." if len(part['text']) > 100 else part['text'])
                                
                                # 提取图像（支持两种命名格式：inline_data 和 inlineData）
                                inline_data_key = None
                                if 'inline_data' in part:
                                    inline_data_key = 'inline_data'
                                elif 'inlineData' in part:
                                    inline_data_key = 'inlineData'
                                
                                if inline_data_key and part[inline_data_key] is not None:
                                    self.log(f"检测到 {inline_data_key} 图像数据")
                                    try:
                                        inline_data = part[inline_data_key]
                                        
                                        # 获取 data 字段（两种格式都使用 'data'）
                                        image_data = inline_data.get('data')
                                        
                                        # 支持两种命名格式获取 mime_type
                                        mime_type = 'image/png'
                                        if 'mime_type' in inline_data:
                                            mime_type = inline_data.get('mime_type', 'image/png')
                                        elif 'mimeType' in inline_data:
                                            mime_type = inline_data.get('mimeType', 'image/png')
                                        
                                        if image_data:
                                            # 直接从 Base64 转换为 Tensor（优化版本）
                                            try:
                                                self.log(f"开始解码 base64 图像数据...")
                                                # 如果 image_data 是字符串格式的 base64，直接使用优化的转换方法
                                                if isinstance(image_data, str):
                                                    self.log(f"[解码方法] 检测到 Base64 字符串，使用优化的转换方法")
                                                    img_tensor = self.base64_to_tensor(image_data)
                                                    images.append(img_tensor)
                                                    self.log(f"成功处理 {inline_data_key} 图像: {img_tensor.shape}")
                                                else:
                                                    # 如果已经是 bytes，需要先转换为 PIL 再转换
                                                    self.log(f"[解码方法] 图像数据是 bytes 格式，使用标准转换方法（asarray）")
                                                    pil_image = Image.open(BytesIO(image_data))
                                                    if pil_image.mode != 'RGB':
                                                        pil_image = pil_image.convert('RGB')
                                                    img_tensor = self.pil2tensor(pil_image)
                                                    images.append(img_tensor)
                                                    self.log(f"成功处理 {inline_data_key} 图像: {img_tensor.shape}")
                                            except Exception as img_error:
                                                self.log(f"处理 {inline_data_key} 图像失败: {str(img_error)}")
                                                traceback.print_exc()
                                                continue
                                        else:
                                            self.log(f"警告: {inline_data_key} 中没有 data 字段")
                                    except Exception as e:
                                        self.log(f"提取 {inline_data_key} 失败: {str(e)}")
                                        traceback.print_exc()
                                        continue
                            
                            if images:
                                # 合并多张图像
                                try:
                                    combined_tensor = torch.cat(images, dim=0)
                                except RuntimeError:
                                    self.log("警告: 图像尺寸不同，返回第一张图像")
                                    combined_tensor = images[0] if images else None
                                
                                if combined_tensor is not None:
                                    total_time = time.time() - function_start_time
                                    self.log(f"[性能] 函数总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text if response_text else '图像生成成功'}"
                                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回 (原生Gemini格式)\n" + formatted_response
                                    return (combined_tensor, full_text)
                            
                            # 调试：检查处理结果
                            self.log(f"[调试] 处理完成 - images数量: {len(images)}, response_text长度: {len(response_text)}")
                            
                            # 如果没有图像但有文本，继续尝试从文本中提取图像数据
                            if response_text:
                                self.log("原生格式返回了文本，尝试从文本中提取图像数据")
                                
                                # 首先尝试提取 base64 数据（data URI）
                                base64_data = self.extract_image_base64(response_text)
                                extracted_images = []
                                
                                if base64_data:
                                    self.log("从响应文本中提取到 Base64 数据")
                                    try:
                                        img_tensor = self.base64_to_tensor(base64_data)
                                        extracted_images.append(img_tensor)
                                        self.log(f"成功解码 Base64 图像: {img_tensor.shape}")
                                    except Exception as b64_error:
                                        self.log(f"解码 Base64 数据失败: {str(b64_error)}")
                                
                                # 如果 base64 提取失败，尝试提取图像URL
                                if not extracted_images:
                                    image_urls = self.extract_image_urls(response_text)
                                    if image_urls:
                                        self.log(f"从文本中提取到 {len(image_urls)} 个图像URL")
                                        failed_urls = []
                                        # 处理从文本中提取的URL
                                        for i, url in enumerate(image_urls):
                                            self.log(f"正在处理图像 {i+1}/{len(image_urls)}: {url[:80]}...")
                                            try:
                                                # 检查是否是 data URI
                                                if url.startswith('data:'):
                                                    self.log(f"检测到 data URI，提取 base64 数据")
                                                    base64_from_uri = self.extract_image_base64(url)
                                                    if base64_from_uri:
                                                        img_tensor = self.base64_to_tensor(base64_from_uri)
                                                        extracted_images.append(img_tensor)
                                                        self.log(f"成功解码 data URI 图像 {i+1}: {img_tensor.shape}")
                                                    else:
                                                        self.log(f"无法从 data URI 中提取 base64 数据")
                                                        failed_urls.append(f"图像 {i+1} (data URI)")
                                                else:
                                                    # 真正的 URL，尝试下载
                                                    self.log(f"正在下载图像 {i+1}/{len(image_urls)}: {url[:80]}...")
                                                    try:
                                                        img_response = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
                                                        img_response.raise_for_status()
                                                        self.log(f"正在处理下载的图像 {i+1}...")
                                                        pil_image = Image.open(BytesIO(img_response.content))
                                                        img_tensor = self.pil2tensor(pil_image)
                                                        extracted_images.append(img_tensor)
                                                        self.log(f"成功下载并转换图像 {i+1}: {img_tensor.shape}")
                                                    except requests.exceptions.Timeout:
                                                        error_msg = f"下载超时 (120秒)"
                                                        self.log(f"处理图像 {i+1} 失败: {error_msg}")
                                                        failed_urls.append(f"图像 {i+1}: {url[:50]}... ({error_msg})")
                                                    except requests.exceptions.RequestException as req_error:
                                                        error_msg = f"网络错误: {str(req_error)}"
                                                        self.log(f"处理图像 {i+1} 失败: {error_msg}")
                                                        failed_urls.append(f"图像 {i+1}: {url[:50]}... ({error_msg})")
                                                    except Exception as download_error:
                                                        error_msg = f"下载或处理错误: {str(download_error)}"
                                                        self.log(f"处理图像 {i+1} 失败: {error_msg}")
                                                        failed_urls.append(f"图像 {i+1}: {url[:50]}... ({error_msg})")
                                            except Exception as img_error:
                                                error_msg = f"处理错误: {str(img_error)}"
                                                self.log(f"处理图像 {i+1} 失败: {error_msg}")
                                                failed_urls.append(f"图像 {i+1}: {url[:50] if not url.startswith('data:') else 'data URI'}... ({error_msg})")
                                                continue
                                        
                                        # 记录下载失败的URL信息
                                        if failed_urls and not extracted_images:
                                            self.log(f"所有图像URL下载都失败了，失败的URL列表: {', '.join(failed_urls)}")
                                        elif failed_urls:
                                            self.log(f"部分图像URL下载失败: {', '.join(failed_urls)}")
                                
                                # 使用提取到的图像
                                if extracted_images:
                                    downloaded_images = extracted_images
                                    try:
                                        combined_tensor = torch.cat(downloaded_images, dim=0)
                                        # 合并成功，直接返回
                                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                        formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text}"
                                        full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回 (原生Gemini格式，从文本URL提取)\n" + formatted_response
                                        return (combined_tensor, full_text)
                                    except RuntimeError:
                                        self.log("警告: 图像尺寸不同，返回第一张图像")
                                        combined_tensor = downloaded_images[0] if downloaded_images else None
                                        
                                        if combined_tensor is not None:
                                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                            formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text}"
                                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回 (原生Gemini格式，从文本URL提取)\n" + formatted_response
                                            return (combined_tensor, full_text)
                                    
                                    # 如果合并失败且没有可用的图像，回退到OpenAI格式
                                    self.log("原生格式文本中的URL下载失败，回退到OpenAI兼容格式")
                                    raise Exception("原生格式URL下载失败")
                                else:
                                    # 原生格式成功但没有图像，回退到OpenAI格式
                                    self.log("原生格式成功但未返回图像，回退到OpenAI兼容格式")
                                    raise Exception("原生格式未返回图像")
                            else:
                                # 原生格式成功但没有内容，回退
                                self.log("原生格式成功但响应为空，回退到OpenAI兼容格式")
                                self.log(f"[调试] 最终状态 - images: {len(images) if 'images' in locals() else 0}, response_text: '{response_text}'")
                                raise Exception("原生格式响应为空")
                        else:
                            self.log("原生格式响应中没有 content.parts，回退到OpenAI兼容格式")
                            raise Exception("原生格式响应格式不正确")
                    else:
                        self.log("原生格式响应中没有 candidates，回退到OpenAI兼容格式")
                        raise Exception("原生格式响应中没有candidates")
                else:
                    # 原生格式请求失败，回退到OpenAI格式
                    error_text = gemini_response.text[:200] if hasattr(gemini_response, 'text') else ""
                    self.log(f"原生 Gemini 格式请求失败 (状态码: {gemini_response.status_code}): {error_text}")
                    raise Exception(f"原生格式请求失败: {gemini_response.status_code}")
                    
            except Exception as gemini_error:
                self.log(f"原生 Gemini 格式失败: {str(gemini_error)}")
                self.log("=" * 50)
                self.log("尝试2: 回退到 OpenAI 兼容格式")
                self.log("=" * 50)
                
                # ========== 尝试2: OpenAI 兼容格式 ==========
                try:
                    # 构建 OpenAI 兼容格式的请求
                    openai_content = []
                    openai_content.append({"type": "text", "text": simple_prompt})
                    
                    # 添加参考图像（使用 image_url 格式）
                    for image_base64 in image_base64_list:
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
                    
                    messages = [{
                        "role": "user",
                        "content": openai_content
                    }]
                    
                    openai_payload = {
                        "model": actual_model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "seed": seed if seed > 0 else None,
                        "max_tokens": 8192
                    }
                    
                    # OpenAI 兼容格式的端点
                    openai_url = f"{actual_base_url.rstrip('/')}/v1/chat/completions"
                    self.log(f"OpenAI 兼容格式 - URL: {openai_url}")
                    
                    # 发送 OpenAI 兼容格式请求
                    openai_response = requests.post(
                        openai_url,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {actual_api_key}"
                        },
                        json=openai_payload,
                        timeout=900
                    )
                    
                    self.log(f"OpenAI 兼容格式 - HTTP响应状态码: {openai_response.status_code}")
                    openai_response.raise_for_status()
                    openai_result = openai_response.json()
                    
                    self.log("✅ OpenAI 兼容格式请求成功")
                    self.log(f"响应结构键: {list(openai_result.keys())}")
                    
                    # 处理 OpenAI 兼容格式的响应
                    images = []
                    response_text = ""
                    
                    # 首先检查是否有 data 字段（OpenAI 图像生成 API 的标准格式）
                    if 'data' in openai_result and isinstance(openai_result['data'], list):
                        self.log(f"检测到 data 字段，包含 {len(openai_result['data'])} 个图像项")
                        data_items = openai_result['data']
                        
                        for i, item in enumerate(data_items):
                            try:
                                # 检查是否有 base64 数据
                                if "b64_json" in item and item["b64_json"]:
                                    self.log(f"处理图像 {i+1}/{len(data_items)}: Base64 数据")
                                    try:
                                        # 使用 base64_to_tensor 方法解码 base64 数据
                                        img_tensor = self.base64_to_tensor(item["b64_json"])
                                        images.append(img_tensor)
                                        self.log(f"成功解码 Base64 图像 {i+1}: {img_tensor.shape}")
                                    except Exception as b64_error:
                                        self.log(f"解码 Base64 图像 {i+1} 失败: {str(b64_error)}")
                                        continue
                                
                                # 检查是否有 URL
                                elif "url" in item and item["url"]:
                                    url = item["url"]
                                    self.log(f"处理图像 {i+1}/{len(data_items)}: URL - {url[:80]}...")
                                    try:
                                        img_response = requests.get(url, timeout=120)
                                        img_response.raise_for_status()
                                        pil_image = Image.open(BytesIO(img_response.content))
                                        img_tensor = self.pil2tensor(pil_image)
                                        images.append(img_tensor)
                                        self.log(f"成功下载并转换图像 {i+1}: {img_tensor.shape}")
                                    except Exception as url_error:
                                        self.log(f"下载图像URL {i+1} 失败: {str(url_error)}")
                                        continue
                            except Exception as item_error:
                                self.log(f"处理图像项 {i+1} 时出错: {str(item_error)}")
                                continue
                    
                    # 如果没有 data 字段，检查 choices 字段（聊天完成格式）
                    elif openai_result.get('choices') and len(openai_result['choices']) > 0:
                        response_text = openai_result["choices"][0]["message"]["content"]
                        self.log(f"API返回文本内容长度: {len(response_text)} 字符")
                        
                        # 尝试从响应文本中提取 base64 数据
                        base64_data = self.extract_image_base64(response_text)
                        if base64_data:
                            self.log("从响应文本中提取到 Base64 数据")
                            try:
                                img_tensor = self.base64_to_tensor(base64_data)
                                images.append(img_tensor)
                                self.log(f"成功解码 Base64 图像: {img_tensor.shape}")
                            except Exception as b64_error:
                                self.log(f"解码 Base64 数据失败: {str(b64_error)}")
                        
                        # 如果 base64 提取失败，尝试提取图像URL
                        if not images:
                            image_urls = self.extract_image_urls(response_text)
                            if image_urls:
                                self.log(f"从响应中提取到 {len(image_urls)} 个图像URL")
                                for i, url in enumerate(image_urls):
                                    self.log(f"正在下载图像 {i+1}/{len(image_urls)}: {url[:80]}...")
                                    try:
                                        img_response = requests.get(url, timeout=120)
                                        img_response.raise_for_status()
                                        self.log(f"正在处理下载的图像 {i+1}...")
                                        pil_image = Image.open(BytesIO(img_response.content))
                                        img_tensor = self.pil2tensor(pil_image)
                                        images.append(img_tensor)
                                        self.log(f"成功下载并转换图像 {i+1}: {img_tensor.shape}")
                                    except Exception as img_error:
                                        self.log(f"处理图像URL {i+1} 失败: {str(img_error)}")
                                        continue
                    
                    # 处理结果
                    if images:
                        # 合并多张图像
                        try:
                            combined_tensor = torch.cat(images, dim=0)
                        except RuntimeError:
                            self.log("警告: 图像尺寸不同，返回第一张图像")
                            combined_tensor = images[0] if images else None
                        
                        if combined_tensor is not None:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            if response_text:
                                formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text}"
                            else:
                                formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n成功生成 {len(images)} 张图像"
                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回 (OpenAI兼容格式)\n" + formatted_response
                            return (combined_tensor, full_text)
                        else:
                            raise Exception("无法合并图像")
                    else:
                        # 没有找到图像，返回参考图像或空白图像
                        error_msg = "没有成功下载或解析任何图像"
                        self.log(error_msg)
                        if response_text:
                            self.log("响应文本中未找到图像数据")
                        if reference_images_count > 0:
                            # 返回第一张参考图像
                            reference_image = all_images[0]
                            formatted_response = f"**User prompt**: {prompt}\n\n**Response**: {response_text if response_text else '未返回文本内容'}"
                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回 (OpenAI兼容格式)\n" + formatted_response + f"\n\n## 注意\n{error_msg}"
                            return (reference_image, full_text)
                        else:
                            # 返回空白图像
                            formatted_response = f"**User prompt**: {prompt}\n\n**Response**: {response_text if response_text else '未返回文本内容'}"
                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回 (OpenAI兼容格式)\n" + formatted_response + f"\n\n## 注意\n{error_msg}"
                            return (self.generate_empty_image(512, 512), full_text)
                        
                except Exception as openai_error:
                    error_message = f"OpenAI兼容格式也失败: {str(openai_error)}"
                    self.log(error_message)
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n两种格式都失败了：\n1. 原生Gemini格式: " + str(gemini_error) + "\n2. OpenAI兼容格式: " + str(openai_error)
                return (self.generate_empty_image(512, 512), full_text)
        
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"Gemini图像生成错误: {str(e)}")
            traceback.print_exc()
            
            # 合并日志和错误信息
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
            return (self.generate_empty_image(512, 512), full_text)

class GeminiImageToPrompt:
    """从图片或视频反推提示词的节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Enter your API key"}),

                "base_url": (["https://wcnb.ai"], {"default": "https://wcnb.ai"}),
                "model_name": ([
                    "gemini-2.5-flash",
                    "gemini-3-pro-preview",
                    "自定义输入 (Custom Input)"
                ], {"default": "gemini-2.5-flash"}),
                "custom_model_name": ("STRING", {"default": "", "multiline": False, "placeholder": "输入自定义模型名称"}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "describe the image"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 100, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "API Respond")
    FUNCTION = "image_to_prompt"
    CATEGORY = "wcnb"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        self.url_file = os.path.join(self.node_dir, "gemini_base_url.txt")
        self.model_file = os.path.join(self.node_dir, "gemini_model_name.txt")
    
    def log(self, message):
        """全局日志函数"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        # 添加控制台输出以便调试
        print(f"[GeminiImageToPrompt] {message}")
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥"""
        # 如果用户输入了有效的密钥，使用并保存
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 显示部分密钥用于调试（前10个字符 + ... + 后4个字符）
            masked_key = f"{user_input_key[:10]}...{user_input_key[-4:]}" if len(user_input_key) > 14 else user_input_key[:10] + "..."
            self.log(f"API密钥: {masked_key}")
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("已保存API密钥到节点目录")
            except Exception as e:
                self.log(f"保存API密钥失败: {e}")
            return user_input_key
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("使用已保存的API密钥")
                    masked_key = f"{saved_key[:10]}...{saved_key[-4:]}" if len(saved_key) > 14 else saved_key[:10] + "..."
                    self.log(f"API密钥: {masked_key}")
                    return saved_key
            except Exception as e:
                self.log(f"读取保存的API密钥失败: {e}")
                
        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥")
        return ""
    
    def get_base_url(self, user_input_url):
        """获取Base URL，优先使用用户输入的URL"""
        # 如果用户输入了有效的URL，使用并保存
        if user_input_url and user_input_url.strip():
            url = user_input_url.strip()
            # 确保URL有协议前缀
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
            self.log(f"使用用户输入的Base URL: {url}")
            # 保存到文件中
            try:
                with open(self.url_file, "w") as f:
                    f.write(url)
                self.log("已保存Base URL到节点目录")
            except Exception as e:
                self.log(f"保存Base URL失败: {e}")
            return url
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.url_file):
            try:
                with open(self.url_file, "r") as f:
                    saved_url = f.read().strip()
                if saved_url:
                    self.log(f"使用已保存的Base URL: {saved_url}")
                    return saved_url
            except Exception as e:
                self.log(f"读取保存的Base URL失败: {e}")
                
        # 默认URL
        default_url = "https://wcnb.ai"
        self.log(f"使用默认Base URL: {default_url}")
        return default_url
    
    def get_model_name(self, user_input_model):
        """获取模型名称，优先使用用户输入的模型"""
        # 如果用户输入了有效的模型名，使用并保存
        if user_input_model and user_input_model.strip():
            model = user_input_model.strip()
            self.log(f"使用用户输入的模型: {model}")
            # 保存到文件中
            try:
                with open(self.model_file, "w") as f:
                    f.write(model)
                self.log("已保存模型名称到节点目录")
            except Exception as e:
                self.log(f"保存模型名称失败: {e}")
            return model
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, "r") as f:
                    saved_model = f.read().strip()
                if saved_model:
                    self.log(f"使用已保存的模型: {saved_model}")
                    return saved_model
            except Exception as e:
                self.log(f"读取保存的模型名称失败: {e}")
                
        # 默认模型
        default_model = "gemini-2.5-flash"
        self.log(f"使用默认模型: {default_model}")
        return default_model
    
    def tensor2pil(self, image: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to PIL image(s)"""
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        if batch_count > 1:
            out = []
            for i in range(batch_count):
                out.extend(self.tensor2pil(image[i]))
            return out

        numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        return [Image.fromarray(numpy_image)]
    
    def _get_video_file_path(self, video):
        """
        Try to extract a filesystem path from a ComfyUI VIDEO object.
        Returns None if it cannot be resolved.
        """
        # VideoFromFile type (private attribute)
        if hasattr(video, "_VideoFromFile__file"):
            path = getattr(video, "_VideoFromFile__file", None)
            if isinstance(path, str) and os.path.exists(path):
                return path

        # Stream-like sources
        if hasattr(video, "get_stream_source"):
            try:
                stream_source = video.get_stream_source()
                if isinstance(stream_source, str) and os.path.exists(stream_source):
                    return stream_source
            except Exception:
                pass

        # Common attributes
        for attr in ("path", "file"):
            if hasattr(video, attr):
                path = getattr(video, attr, None)
                if isinstance(path, str) and os.path.exists(path):
                    return path

        return None

    def encode_video_b64(self, video):
        """
        Encode ComfyUI VIDEO object to base64 MP4 bytes with compression.
        Apply ffmpeg compression to reduce base64 size.
        """
        video_path = self._get_video_file_path(video)
        temp_original = None
        
        # If no path, save to temp file first
        if not video_path:
            if hasattr(video, "save_to"):
                temp_original = f"temp_video_original_{time.time()}.mp4"
                try:
                    video.save_to(temp_original)
                    video_path = temp_original
                except Exception as e:
                    self.log(f"Error saving video: {str(e)}")
                    raise ValueError(f"Unable to save video: {str(e)}")
            else:
                raise ValueError(f"Unable to read video data from object type: {type(video)}")
        
        # Get original video info
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,duration',
                '-of', 'json',
                video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                probe_data = json.loads(probe_result.stdout)
                if 'streams' in probe_data and len(probe_data['streams']) > 0:
                    stream = probe_data['streams'][0]
                    width = stream.get('width', 0)
                    height = stream.get('height', 0)
                    duration = float(stream.get('sduration', 0))
                    self.log(f"Original video: {width}x{height}, {duration:.1f}s")
        except Exception as e:
            self.log(f"Could not probe video: {e}")
        
        # Get original file size
        try:
            original_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            self.log(f"Original video file size: {original_size_mb:.2f}MB")
        except:
            original_size_mb = 0
        
        # Compress video using ffmpeg
        compressed_path = f"temp_video_compressed_{time.time()}.mp4"
        
        try:
            # Compression strategy:
            # 1. Extract only first 5 seconds (sufficient for analysis)
            # 2. Limit resolution to 720p max (1280x720)
            # 3. Lower bitrate to 400k
            # 4. Reduce frame rate to 10fps
            # 5. Use fast encoding preset
            compress_cmd = [
                'ffmpeg', '-i', video_path,
                '-t', '5',  # Only first 5 seconds
                '-vf', 'scale=\'min(1280,iw)\':-2',  # Max width 1280, keep aspect ratio
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '30',  # Higher CRF = more compression
                '-b:v', '400k',  # Limit video bitrate
                '-maxrate', '400k',
                '-bufsize', '800k',
                '-r', '10',  # 10 fps (reduced for smaller size)
                '-an',  # Remove audio to save size
                '-y',  # Overwrite output
                compressed_path
            ]
            
            self.log(f"Compressing video (first 5s only) with ffmpeg...")
            result = subprocess.run(compress_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                self.log(f"FFmpeg compression failed: {result.stderr}")
                # Fallback: use original video
                final_path = video_path
            else:
                compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                self.log(f"Compressed video size: {compressed_size_mb:.2f}MB (reduced {((original_size_mb - compressed_size_mb) / original_size_mb * 100):.1f}%)")
                final_path = compressed_path
                
        except FileNotFoundError:
            self.log(f"Warning: ffmpeg not found, using original video without compression")
            final_path = video_path
        except subprocess.TimeoutExpired:
            self.log(f"Warning: ffmpeg timeout, using original video")
            final_path = video_path
        except Exception as e:
            self.log(f"Warning: compression failed ({str(e)}), using original video")
            final_path = video_path
        
        # Read and encode to base64
        try:
            with open(final_path, "rb") as f:
                video_bytes = f.read()
                base64_data = base64.b64encode(video_bytes).decode("utf-8")
                
            base64_size_mb = len(base64_data) / (1024 * 1024)
            self.log(f"Final video base64 size: {base64_size_mb:.2f}MB")
            
            if base64_size_mb > 10.0:
                self.log(f"Warning: Base64 size is very large ({base64_size_mb:.2f}MB), may cause API issues")
            
            return base64_data
            
        finally:
            # Cleanup temp files
            try:
                if temp_original and os.path.exists(temp_original):
                    os.remove(temp_original)
                if os.path.exists(compressed_path):
                    os.remove(compressed_path)
            except Exception:
                pass

    def image_to_prompt(self, api_key, base_url, model_name, custom_model_name, role, prompt, temperature, seed, ref_image=None, video=None):
        """从图片或视频反推提示词"""
        response_text = ""
        self.log_messages = []
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            
            # 处理Base URL
            actual_base_url = self.get_base_url(base_url)
            
            # 处理模型名称：如果选择了"自定义输入"，使用 custom_model_name；否则使用选择的预设值
            if model_name == "自定义输入 (Custom Input)":
                user_model_name = custom_model_name.strip() if custom_model_name and custom_model_name.strip() else "gemini-2.5-flash"
                if not user_model_name or user_model_name == "":
                    user_model_name = "gemini-2.5-flash"
                    self.log("警告: 自定义模型名称为空，使用默认值 gemini-2.5-flash")
                self.log(f"使用自定义模型: {user_model_name}")
            else:
                user_model_name = model_name if model_name else "gemini-2.5-flash"
                self.log(f"使用选择的模型: {user_model_name}")
            
            # 使用 get_model_name 方法获取模型名称（支持从文件读取）
            actual_model_name = self.get_model_name(user_model_name)
            
            # 保存原始模型名称用于日志
            original_model_name = actual_model_name
            
            # 根据不同的 API 处理模型名称格式
            # 对于 Google 官方 API，需要添加 models/ 前缀（如果不存在）
            if "generativelanguage.googleapis.com" in actual_base_url:
                if not actual_model_name.startswith("models/"):
                    actual_model_name = f"models/{actual_model_name}"
                    self.log(f"为 Google API 添加 models/ 前缀: {original_model_name} -> {actual_model_name}")
            # 对于 wcnb.ai，确保去掉 models/ 前缀（如果存在）
            elif "wcnb.ai" in actual_base_url:
                if actual_model_name.startswith("models/"):
                    actual_model_name = actual_model_name.replace("models/", "", 1)
                    self.log(f"为 wcnb.ai 去掉 models/ 前缀: {original_model_name} -> {actual_model_name}")
            
            # 添加调试日志
            self.log(f"最终配置 - URL: {actual_base_url}, 模型: {actual_model_name}")
            
            # 处理温度值：如果没有手动输入，使用默认值 0.6
            actual_temperature = temperature if temperature is not None else 0.6
            
            # 处理种子值：如果没有手动输入，使用默认值 100
            actual_seed = seed if seed is not None else 100
            
            self.log(f"温度: {actual_temperature}, 种子值: {actual_seed}")
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message
                return ("", full_text)
            
            # Priority: video > image > text (align with Comfly_LLm_API behavior)
            # 使用用户输入的 role 和 prompt
            user_prompt = prompt.strip() if prompt and prompt.strip() else "describe the image"
            
            # 处理视频输入（优先级最高）
            if video is not None:
                try:
                    self.log(f"Processing video input...")
                    base64_video = self.encode_video_b64(video)
                    
                    # Log base64 info (truncated)
                    base64_preview = base64_video[:100] + f"...[total {len(base64_video)} chars]"
                    self.log(f"Video base64 preview: {base64_preview}")
                    self.log(f"Sending video to model: {actual_model_name}")
                    
                    # 构建消息内容 - 使用 OpenAI 兼容格式
                    message_content = [
                        {
                            "type": "text",
                            "text": f"{user_prompt}"
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{base64_video}"
                            }
                        },
                    ]
                    
                    messages = [
                        {'role': 'system', 'content': f'{role}'},
                        {'role': 'user', 'content': message_content},
                    ]
                    
                except Exception as e:
                    error_msg = f"Error encoding video: {str(e)}"
                    self.log(error_msg)
                    import traceback
                    traceback.print_exc()
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_msg
                    return ("", full_text)
            
            # 处理图片输入
            elif ref_image is not None:
                try:
                    self.log(f"Processing image input...")
                    pil_image = self.tensor2pil(ref_image)[0]
                    img_byte_arr = BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    image_bytes = img_byte_arr.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    self.log(f"Image base64 size: {len(image_base64) / (1024*1024):.2f}MB")
                    
                    # 构建消息内容 - 使用 OpenAI 兼容格式
                    message_content = [
                        {
                            "type": "text",
                            "text": f"{user_prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                    ]
                    
                    messages = [
                        {'role': 'system', 'content': f'{role}'},
                        {'role': 'user', 'content': message_content},
                    ]
                    
                except Exception as img_error:
                    error_msg = f"Error encoding image: {str(img_error)}"
                    self.log(error_msg)
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_msg
                    return ("", full_text)
            
            # 纯文本输入
            else:
                messages = [
                    {'role': 'system', 'content': f'{role}'},
                    {'role': 'user', 'content': f'{user_prompt}'},
                ]
            
            # 准备 OpenAI 兼容格式的消息内容
            openai_message_content = []
            for msg in messages:
                if msg['role'] == 'user':
                    if isinstance(msg['content'], str):
                        openai_message_content.append({
                            "type": "text",
                            "text": msg['content']
                        })
                    elif isinstance(msg['content'], list):
                        openai_message_content.extend(msg['content'])
            
            # 准备 Gemini 格式的内容
            contents = []
            parts = []
            system_content = None
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_content = msg['content']
                elif msg['role'] == 'user':
                    if isinstance(msg['content'], str):
                        parts.append({"text": msg['content']})
                    elif isinstance(msg['content'], list):
                        for item in msg['content']:
                            if item.get('type') == 'text':
                                parts.append({"text": item['text']})
                            elif item.get('type') == 'image_url':
                                url = item['image_url']['url']
                                if url.startswith('data:'):
                                    mime_type, base64_data = url.split(',', 1)
                                    mime_type = mime_type.split(';')[0].split(':')[1]
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": base64_data
                                        }
                                    })
                            elif item.get('type') == 'video_url':
                                url = item['video_url']['url']
                                if url.startswith('data:'):
                                    mime_type, base64_data = url.split(',', 1)
                                    mime_type = mime_type.split(';')[0].split(':')[1]
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": base64_data
                                        }
                                    })
            
            # 如果有 system content，合并到第一个 user message
            if system_content and parts:
                if parts[0].get('text'):
                    parts[0]['text'] = f"{system_content}\n\n{parts[0]['text']}"
                else:
                    parts.insert(0, {"text": system_content})
            
            contents = [{
                "role": "user",
                "parts": parts
            }]
            
            # 构建API配置列表 - 智能尝试多种格式
            api_configs = []
            
            # 配置1: OpenAI 兼容格式 (chat/completions) - 最常用，优先尝试
            processed_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    # 将 system role 合并到 user message
                    system_content = msg['content']
                elif msg['role'] == 'user':
                    if system_content:
                        if isinstance(msg['content'], str):
                            msg['content'] = f"{system_content}\n\n{msg['content']}"
                        elif isinstance(msg['content'], list):
                            text_items = [item for item in msg['content'] if item.get('type') == 'text']
                            other_items = [item for item in msg['content'] if item.get('type') != 'text']
                            if text_items:
                                text_items[0]['text'] = f"{system_content}\n\n{text_items[0]['text']}"
                                msg['content'] = text_items + other_items
                            else:
                                msg['content'] = [{"type": "text", "text": system_content}] + msg['content']
                    processed_messages.append(msg)
            
            api_configs.append({
                "name": "OpenAI Compatible (chat/completions)",
                "url": f"{actual_base_url}/v1/chat/completions",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {actual_api_key}"
                },
                "data": {
                    "model": actual_model_name,
                    "messages": processed_messages,
                    "temperature": actual_temperature,
                    "max_tokens": 2048
                },
                "response_type": "openai"
            })
            
            # 只有当 seed 是有效值（非负数）时才添加到 OpenAI 配置
            if actual_seed is not None and actual_seed >= 0:
                api_configs[0]["data"]["seed"] = actual_seed
            
            # 配置2: Google Gemini 官方 API 格式（如果是官方API，优先使用）
            if "generativelanguage.googleapis.com" in actual_base_url:
                generation_config = {
                    "temperature": actual_temperature,
                    "maxOutputTokens": 2048
                }
                if actual_seed is not None and actual_seed >= 0:
                    generation_config["seed"] = actual_seed
                
                api_configs_gemini = [{
                    "name": "Google Gemini API",
                    "url": f"{actual_base_url}/v1beta/{actual_model_name}:generateContent",
                    "headers": {
                        "Content-Type": "application/json",
                        "x-goog-api-key": actual_api_key
                    },
                    "data": {
                        "contents": contents,
                        "generationConfig": generation_config
                    },
                    "response_type": "gemini"
                }]
                # 将 Gemini 配置插入到最前面
                api_configs = api_configs_gemini + api_configs
            
            # 配置3: Gemini 格式 (通用端点) - 作为备选
            generation_config = {
                "temperature": actual_temperature,
                "maxOutputTokens": 2048
            }
            if actual_seed is not None and actual_seed >= 0:
                generation_config["seed"] = actual_seed
            
            api_configs.append({
                "name": "Gemini Format (通用)",
                "url": f"{actual_base_url}/v1beta/{actual_model_name}:generateContent",
                "headers": {
                    "Content-Type": "application/json",
                    "x-goog-api-key": actual_api_key
                },
                "data": {
                    "contents": contents,
                    "generationConfig": generation_config
                },
                "response_type": "gemini"
            })
            
            # 尝试不同的 API 配置，直到成功
            response_json = None
            last_error = None
            successful_config = None
            input_type = "视频" if video is not None else ("图片" if ref_image is not None else "文本")
            
            for config in api_configs:
                try:
                    self.log(f"尝试 {config['name']} 格式，URL: {config['url']}")
                    
                    # 发送HTTP请求
                    try:
                        response = requests.post(
                            config['url'],
                            headers=config['headers'],
                            json=config['data'],
                            timeout=120
                        )
                    except Exception as e:
                        self.log(f"请求异常: {str(e)}")
                        last_error = f"{config['name']} 请求失败: {str(e)}"
                        continue
                    
                    self.log(f"HTTP响应状态码: {response.status_code}")
                    
                    # 检查是否是 HTML 响应（说明端点不存在）
                    response_text = response.text if hasattr(response, 'text') else ""
                    if response_text.strip().startswith('<!') or response_text.strip().startswith('<html'):
                        self.log(f"{config['name']} 返回HTML页面，端点可能不存在，跳过")
                        last_error = f"{config['name']} 端点不存在（返回HTML页面）"
                        continue
                    
                    if response.status_code == 200:
                        try:
                            if response_text and response_text.strip():
                                response_json = response.json()
                                successful_config = config
                                self.log(f"✅ 成功使用 {config['name']} 格式")
                                break
                            else:
                                self.log(f"{config['name']} 返回空响应，尝试下一个格式")
                                continue
                        except json.JSONDecodeError as e:
                            self.log(f"{config['name']} 响应不是有效JSON: {str(e)}")
                            response_preview = response_text[:200] if response_text else "响应为空"
                            self.log(f"响应内容预览: {response_preview}")
                            last_error = f"{config['name']} 格式失败: 响应不是有效JSON"
                            continue
                    elif response.status_code == 404:
                        self.log(f"{config['name']} 端点不存在 (404)")
                        last_error = f"{config['name']} 端点不存在 (404)"
                        continue
                    else:
                        error_text = response_text[:200] if response_text else "无响应内容"
                        self.log(f"{config['name']} 返回状态码 {response.status_code}: {error_text}")
                        last_error = f"{config['name']} 格式失败: 状态码 {response.status_code}"
                        continue
                        
                except Exception as e:
                    self.log(f"{config['name']} 处理异常: {str(e)}")
                    last_error = f"{config['name']} 处理失败: {str(e)}"
                    continue
            
            # 如果所有配置都失败
            if response_json is None:
                error_msg = f"所有API格式尝试均失败。\n\n最后错误: {last_error}\n\n"
                error_msg += "## 可能的原因和解决方案：\n"
                error_msg += "1. API端点不正确：该服务可能使用不同的API端点格式\n"
                error_msg += "   - 检查服务文档确认正确的端点路径\n"
                error_msg += "2. 模型名称不正确：当前使用的模型可能在该API服务上不可用\n"
                error_msg += "   - 尝试使用其他模型名称\n"
                error_msg += "3. API密钥或权限问题：确认API密钥有效且有相应权限\n"
                error_msg += "4. 服务暂时不可用：某些端点可能暂时不可用\n\n"
                error_msg += "## 建议：\n"
                error_msg += "- 查看上方的详细日志了解每个格式的失败原因\n"
                self.log(error_msg)
                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_msg
                return ("", full_text)
            
            self.log("API响应接收成功，正在处理...")
            
            # 添加调试输出：打印响应结构
            self.log(f"响应JSON结构: {list(response_json.keys()) if isinstance(response_json, dict) else type(response_json)}")
            if isinstance(response_json, dict):
                import json
                response_str = json.dumps(response_json, ensure_ascii=False, indent=2)
                self.log(f"完整响应内容（前500字符）: {response_str[:500]}")
            
            # 根据成功的配置类型处理不同的响应格式
            response_type = successful_config.get('response_type', 'unknown')
            self.log(f"使用响应类型: {response_type}")
            
            # 提取文本响应 - 根据响应类型解析
            if response_type == "openai":
                # OpenAI 兼容格式解析
                if response_json.get('choices') and len(response_json['choices']) > 0:
                    message = response_json['choices'][0].get('message', {})
                    # 检查 content 字段（可能是字符串或 None）
                    content = message.get('content')
                    if content is not None and content != "":
                        response_text = content
                        self.log(f"API返回文本: {response_text[:200]}..." if len(response_text) > 200 else response_text)
                    else:
                        # content 为空，尝试从其他字段获取
                        # 某些 API 可能将内容放在其他字段
                        if message.get('text'):
                            response_text = message['text']
                        elif message.get('response'):
                            response_text = message['response']
                        else:
                            # 检查是否有 reasoning 或其他字段
                            if 'reasoning' in message:
                                response_text = message['reasoning']
                            elif 'output' in message:
                                response_text = message['output']
                            else:
                                self.log(f"警告: message.content 为空。message 字段: {list(message.keys())}")
                                # 检查是否有其他 choices
                                if len(response_json['choices']) > 1:
                                    for i, choice in enumerate(response_json['choices']):
                                        alt_content = choice.get('message', {}).get('content', '')
                                        if alt_content:
                                            response_text = alt_content
                                            self.log(f"从 choice[{i}] 获取内容")
                                            break
                elif response_json.get('content'):
                    # 兼容其他格式
                    content_data = response_json['content']
                    if isinstance(content_data, list):
                        for item in content_data:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                response_text += item.get('text', '')
                    elif isinstance(content_data, str):
                        response_text = content_data
                else:
                    self.log(f"警告: 未找到预期的响应字段。可用字段: {list(response_json.keys())}")
            
            elif response_type == "gemini":
                # Gemini API 格式解析
                if not response_json.get('candidates'):
                    self.log("API响应中没有candidates")
                    full_text = "\n".join(self.log_messages) + "\n\nAPI返回了空响应"
                    return ("", full_text)
                
                for part in response_json['candidates'][0]['content']['parts']:
                    if 'text' in part and part['text'] is not None:
                        response_text += part['text']
                        self.log(f"API返回文本: {response_text[:100]}..." if len(response_text) > 100 else response_text)
            
            if not response_text or response_text.strip() == "":
                # 检查是否有 tokens 但无内容（可能是 API 问题）
                usage = response_json.get('usage', {})
                completion_tokens = usage.get('completion_tokens', 0)
                if completion_tokens > 0:
                    error_msg = f"API返回了 {completion_tokens} 个 tokens，但 content 字段为空。"
                    error_msg += "\n这可能是因为："
                    error_msg += "\n1. API 返回格式异常"
                    error_msg += "\n2. 内容可能在流式响应中（需要启用流式处理）"
                    error_msg += "\n3. API 服务配置问题"
                    self.log(error_msg)
                    # 尝试从响应中提取任何可能的文本内容
                    response_str = json.dumps(response_json, ensure_ascii=False, indent=2)
                    self.log(f"完整响应内容: {response_str[:1000]}...")
                    response_text = f"错误: API返回了 tokens 但 content 为空。请检查 API 服务配置。\n\n响应详情:\n{response_str[:500]}"
                else:
                    response_text = "API未返回任何文本描述"
                    self.log("警告: 未能从API响应中提取到文本描述")
            
            # 合并日志和API返回文本
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 生成的提示词\n" + response_text
            
            return (response_text, full_text)
        
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"图片转提示词错误: {str(e)}")
            
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
            return ("", full_text)


# 简单的视频适配器类
class SimpleVideoAdapter:
    def __init__(self, video_path_or_url):
        if video_path_or_url and video_path_or_url.startswith('http'):
            self.is_url = True
            self.video_url = video_path_or_url
            self.video_path = None
        else:
            self.is_url = False
            self.video_path = video_path_or_url
            self.video_url = None
    
    def get_dimensions(self):
        """获取视频尺寸，返回 (width, height)"""
        if self.is_url:
            # 对于URL视频，返回默认尺寸
            return 1280, 720
        else:
            # 对于本地文件，尝试使用OpenCV获取实际尺寸
            try:
                import cv2
                if self.video_path and os.path.exists(self.video_path):
                    cap = cv2.VideoCapture(self.video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    return width, height
            except Exception as e:
                print(f"Error getting video dimensions: {str(e)}")
            # 如果无法获取，返回默认尺寸
            return 1280, 720
    
    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        """保存视频到指定路径"""
        if self.is_url:
            try:
                response = requests.get(self.video_url, stream=True)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e:
                print(f"Error downloading video from URL: {str(e)}")
                return False
        else:
            try:
                import shutil
                if self.video_path and os.path.exists(self.video_path):
                    shutil.copyfile(self.video_path, output_path)
                    return True
            except Exception as e:
                print(f"Error saving video: {str(e)}")
            return False

class soraPromptToVideo:
    """使用 sora-2 模型从文字生成视频的节点，使用异步轮询和流式响应"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Enter your API key"}),
                "base_url": (["https://wcnb.ai"], {"default": "https://wcnb.ai"}),
                "model_name": ([
                    "sora-2",
                    "sora-2-pro",
                    "自定义输入 (Custom Input)"
                ], {"default": "sora-2"}),
                "custom_model_name": ("STRING", {"default": "", "multiline": False, "placeholder": "输入自定义模型名称"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "size": (["1280x720", "720x1280"], {"default": "720x1280"}),
                "seconds": (["10", "15", "25"], {"default": "10"}),
                "input_reference": ("IMAGE",),
                "watermark": ("BOOLEAN", {"default": False}),
                "private": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "API Respond")
    FUNCTION = "generate_video"
    CATEGORY = "wcnb"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        self.url_file = os.path.join(self.node_dir, "gemini_base_url.txt")
        self.model_file = os.path.join(self.node_dir, "gemini_model_name.txt")
        self.timeout = 900
    
    def log(self, message):
        """全局日志函数"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        print(f"[soraPromptToVideo] {message}")
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥"""
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            masked_key = f"{user_input_key[:10]}...{user_input_key[-4:]}" if len(user_input_key) > 14 else user_input_key[:10] + "..."
            self.log(f"API密钥: {masked_key}")
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("已保存API密钥到节点目录")
            except Exception as e:
                self.log(f"保存API密钥失败: {e}")
            return user_input_key
            
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("使用已保存的API密钥")
                    masked_key = f"{saved_key[:10]}...{saved_key[-4:]}" if len(saved_key) > 14 else saved_key[:10] + "..."
                    self.log(f"API密钥: {masked_key}")
                    return saved_key
            except Exception as e:
                self.log(f"读取保存的API密钥失败: {e}")
                
        self.log("警告: 未提供有效的API密钥")
        return ""
    
    def get_base_url(self, user_input_url):
        """获取Base URL，优先使用用户输入的URL"""
        # 如果用户输入了有效的URL，使用并保存
        if user_input_url and user_input_url.strip():
            url = user_input_url.strip()
            # 确保URL有协议前缀
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
            self.log(f"使用用户输入的Base URL: {url}")
            # 保存到文件中
            try:
                with open(self.url_file, "w") as f:
                    f.write(url)
                self.log("已保存Base URL到节点目录")
            except Exception as e:
                self.log(f"保存Base URL失败: {e}")
            return url
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.url_file):
            try:
                with open(self.url_file, "r") as f:
                    saved_url = f.read().strip()
                if saved_url:
                    self.log(f"使用已保存的Base URL: {saved_url}")
                    return saved_url
            except Exception as e:
                self.log(f"读取保存的Base URL失败: {e}")
                
        # 默认URL
        default_url = "https://wcnb.ai"
        self.log(f"使用默认Base URL: {default_url}")
        return default_url
    
    def tensor2pil(self, image: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to PIL image(s)"""
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        if batch_count > 1:
            out = []
            for i in range(batch_count):
                out.extend(self.tensor2pil(image[i]))
            return out

        numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        return [Image.fromarray(numpy_image)]
    
    def image_to_base64(self, image):
        """Convert ComfyUI IMAGE tensor to base64 string with data URI prefix"""
        if image is None:
            return None
        
        try:
            pil_image = self.tensor2pil(image)[0]
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            image_bytes = img_byte_arr.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            self.log(f"Error encoding image: {str(e)}")
            return None
    
    def image_to_file_tuple(self, image):
        """Convert ComfyUI IMAGE tensor to file tuple for multipart/form-data upload
        Returns: (filename, file_content, content_type) tuple or None
        """
        if image is None:
            return None
        
        try:
            pil_image = self.tensor2pil(image)[0]
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            file_content = img_byte_arr.read()
            return ('image.png', file_content, 'image/png')
        except Exception as e:
            self.log(f"Error converting image to file tuple: {str(e)}")
            return None
    
    def download_and_convert_video(self, video_url: str) -> Optional[Any]:
        """
        下载视频URL并转换为ComfyUI VIDEO对象，参考 Comfly.py 的实现
        - 校验URL合法性
        - 下载视频到临时文件
        - 转换为ComfyUI VIDEO对象
        - 出错返回 None，保证节点稳定
        """
        try:
            if not video_url or not isinstance(video_url, str):
                self.log(f"无效的视频URL: {video_url}")
                return None
            if not video_url.startswith(("http://", "https://")):
                self.log(f"不支持的URL格式: {video_url}")
                return None

            self.log(f"🎬 开始下载视频: {video_url[:80]}...")
            
            # 获取临时目录
            if HAS_FOLDER_PATHS:
                temp_dir = folder_paths.get_temp_directory()
            else:
                temp_dir = tempfile.gettempdir()
            
            # 创建临时文件
            timestamp = int(time.time())
            temp_filename = f"sora_video_{timestamp}.mp4"
            temp_filepath = os.path.join(temp_dir, temp_filename)
            
            try:
                # 下载视频
                response = requests.get(video_url, stream=True, timeout=120)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(temp_filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                if int(progress) % 10 == 0:  # 每10%打印一次
                                    self.log(f"下载进度: {progress:.1f}%")
                
                file_size_mb = os.path.getsize(temp_filepath) / (1024 * 1024)
                self.log(f"✅ 视频下载完成，文件大小: {file_size_mb:.2f}MB")
                
                # 尝试转换为ComfyUI VIDEO对象
                if HAS_VIDEO_FROM_FILE and os.path.exists(temp_filepath):
                    try:
                        video_output = VideoFromFile(temp_filepath)
                        self.log(f"✅ 视频已转换为ComfyUI VIDEO对象")
                        return video_output
                    except Exception as e:
                        self.log(f"⚠️ 无法转换为VideoFromFile，使用SimpleVideoAdapter: {str(e)}")
                        # 回退到SimpleVideoAdapter，但使用本地文件路径
                        return SimpleVideoAdapter(temp_filepath)
                else:
                    # 使用本地文件路径创建适配器
                    self.log(f"✅ 使用本地文件路径创建视频适配器")
                    return SimpleVideoAdapter(temp_filepath)
                    
            except requests.exceptions.RequestException as download_error:
                self.log(f"❌ 视频下载失败: {download_error}")
                # 清理可能的部分下载文件
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except:
                        pass
                return None
            except Exception as e:
                self.log(f"❌ 视频处理失败: {str(e)}")
                # 清理可能的部分下载文件
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except:
                        pass
                return None
                
        except Exception as e:
            self.log(f"视频下载转换过程出错: {e}")
            traceback.print_exc()
            return None
    
    def _try_async_call(self, api_key, base_url, model_name, prompt, input_reference, seed, size, seconds, watermark, private, pbar):
        """异步调用：不使用流式响应，直接提交任务然后轮询"""
        try:
            self.log("🔄 开始异步调用模式...")
            pbar.update_absolute(25)
            
            # 调试日志：检查模型名称
            self.log(f"[调试] 异步调用 - 接收到的模型名称: '{model_name}'")
            self.log(f"[调试] 异步调用 - 模型名称类型: {type(model_name)}")
            self.log(f"[调试] 异步调用 - 模型名称是否为空: {not model_name}")
            self.log(f"[调试] 异步调用 - 模型名称是否为None: {model_name is None}")
            self.log(f"[调试] 异步调用 - 模型名称长度: {len(str(model_name)) if model_name else 0}")
            
            # 验证模型名称
            if not model_name or model_name == "" or model_name is None:
                error_message = f"错误: 异步调用时模型名称为空。接收到的值: {repr(model_name)}"
                self.log(error_message)
                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
                return (SimpleVideoAdapter(""), full_text)
            
            # 验证 prompt
            if not prompt or (isinstance(prompt, str) and prompt.strip() == ""):
                error_message = "错误: prompt 不能为空，请输入视频生成的提示词。"
                self.log(error_message)
                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
                return (SimpleVideoAdapter(""), full_text)
            
            # 构建表单数据 - 使用 multipart/form-data 格式
            data = {
                "model": model_name,
                "prompt": prompt,
                "size": size,
                "seconds": seconds,  # 保持字符串格式
                "watermark": str(watermark).lower(),  # 转换为字符串 "true"/"false"
                "private": str(private).lower()  # 转换为字符串 "true"/"false"
            }
            
            if seed is not None and seed >= 0:
                data["seed"] = str(seed)
            
            # 处理参考图片 - 转换为文件对象
            files = {}
            if input_reference is not None:
                file_tuple = self.image_to_file_tuple(input_reference)
                if file_tuple:
                    files["input_reference"] = file_tuple
                    self.log("成功准备参考图片文件用于上传")
            
            # 调试日志：检查数据
            self.log(f"[调试] 异步调用 - data中的model字段: '{data.get('model')}'")
            self.log(f"[调试] 异步调用 - data完整内容: {json.dumps(data, ensure_ascii=False)}")
            self.log(f"[调试] 异步调用 - files中包含的字段: {list(files.keys())}")
            
            # 构建API URL
            api_url = f"{base_url}/v1/videos"
            headers = {
                "Authorization": f"Bearer {api_key}"
                # 不设置 Content-Type，让 requests 自动设置为 multipart/form-data
            }
            
            self.log(f"异步请求API，URL: {api_url}")
            self.log(f"异步请求API，使用 multipart/form-data 格式")
            pbar.update_absolute(30)
            
            # 发送异步HTTP请求 - 使用 multipart/form-data
            try:
                self.log(f"异步请求API，api_url: {api_url}")
                response = requests.post(
                    api_url,
                    headers=headers,
                    data=data,
                    files=files if files else None,
                    timeout=self.timeout
                    
                )
            except Exception as e:
                error_msg = f"异步API请求失败: {str(e)}"
                self.log(error_msg)
                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_msg
                return (SimpleVideoAdapter(""), full_text)
            
            self.log(f"异步HTTP响应状态码: {response.status_code}")
            
            # 调试日志：检查响应头
            self.log(f"[调试] 异步响应 - Content-Type: {response.headers.get('Content-Type', 'N/A')}")
            self.log(f"[调试] 异步响应 - 响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                error_msg = f"异步API请求失败，状态码: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f"，错误详情: {error_detail}"
                except:
                    error_msg += f"，响应内容: {response.text[:200]}"
                self.log(error_msg)
                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_msg
                return (SimpleVideoAdapter(""), full_text)
            
            # 检查响应体是否为空
            response_text = response.text
            response_content = response.content
            
            self.log(f"[调试] 异步响应 - 响应体长度: {len(response_content)} 字节")
            self.log(f"[调试] 异步响应 - 响应文本长度: {len(response_text)} 字符")
            self.log(f"[调试] 异步响应 - 响应文本内容: {response_text[:500]}")
            
            if not response_text or len(response_text.strip()) == 0:
                # 响应体为空，尝试从响应头中提取信息
                location = response.headers.get('Location', '')
                task_id_header = response.headers.get('X-Task-Id', '') or response.headers.get('Task-Id', '')
                
                self.log(f"[调试] 异步响应 - Location头: {location}")
                self.log(f"[调试] 异步响应 - Task-Id头: {task_id_header}")
                
                if task_id_header:
                    task_id = task_id_header
                    self.log(f"从响应头中提取到task_id: {task_id}")
                    # 继续执行轮询逻辑
                elif location:
                    # 尝试从Location中提取task_id
                    location_match = re.search(r'/([^/]+)/?$', location)
                    if location_match:
                        task_id = location_match.group(1)
                        self.log(f"从Location头中提取到task_id: {task_id}")
                        # 继续执行轮询逻辑
                    else:
                        error_message = "异步调用成功，但响应体为空且无法从响应头中提取task_id"
                        self.log(error_message)
                        full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message + f"\n\n响应头: {dict(response.headers)}"
                        return (SimpleVideoAdapter(""), full_text)
                else:
                    error_message = "异步调用成功，但响应体为空且响应头中无task_id信息"
                    self.log(error_message)
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message + f"\n\n响应头: {dict(response.headers)}"
                    return (SimpleVideoAdapter(""), full_text)
            else:
                # 解析异步响应
                try:
                    response_json = response.json()
                    self.log(f"[调试] 异步响应 - JSON结构: {list(response_json.keys()) if isinstance(response_json, dict) else type(response_json)}")
                    self.log(f"[调试] 异步响应 - JSON完整内容: {json.dumps(response_json, ensure_ascii=False, indent=2)[:1000]}")
                    
                    full_response_text = ""
                    task_id = None
                    data_preview_url = None
                    video_url = None
                    
                    # 格式1: 直接包含 task_id 或 id 字段
                    if 'task_id' in response_json:
                        task_id = response_json.get('task_id')
                        video_url = response_json.get('video_url') or response_json.get('url')
                        self.log(f"从JSON中直接获取: task_id={task_id}, video_url={video_url}")
                        full_response_text = json.dumps(response_json, ensure_ascii=False)
                    elif 'id' in response_json:
                        task_id = response_json.get('id')
                        video_url = response_json.get('video_url') or response_json.get('url')
                        self.log(f"从JSON中直接获取: id={task_id}, video_url={video_url}")
                        full_response_text = json.dumps(response_json, ensure_ascii=False)
                    
                    # 格式2: 包含 data 字段
                    elif 'data' in response_json:
                        data = response_json.get('data', {})
                        if isinstance(data, dict):
                            task_id = data.get('task_id') or data.get('id')
                            video_url = data.get('video_url') or data.get('url')
                            full_response_text = json.dumps(data, ensure_ascii=False)
                            self.log(f"从data字段提取: task_id={task_id}, video_url={video_url}")
                    
                    # 格式3: OpenAI chat completions 格式 (choices)
                    elif 'choices' in response_json and response_json['choices']:
                        message = response_json['choices'][0].get('message', {})
                        full_response_text = message.get('content', '')
                        self.log(f"从choices格式提取内容，长度: {len(full_response_text)}")
                    
                    # 格式4: 包含 result 字段
                    elif 'result' in response_json:
                        result = response_json.get('result', {})
                        if isinstance(result, dict):
                            task_id = result.get('task_id') or result.get('id')
                            video_url = result.get('video_url') or result.get('url')
                        full_response_text = json.dumps(result, ensure_ascii=False)
                        self.log(f"从result字段提取: task_id={task_id}, video_url={video_url}")
                    
                    # 格式5: 如果都没有，使用完整响应
                    else:
                        full_response_text = json.dumps(response_json, ensure_ascii=False)
                        self.log(f"使用完整响应作为内容")
                        # 尝试从完整响应中提取字段
                        if isinstance(response_json, dict):
                            task_id = response_json.get('task_id') or response_json.get('id')
                            video_url = response_json.get('video_url') or response_json.get('url')
                    
                    self.log(f"异步响应接收成功，内容长度: {len(full_response_text)}, task_id: {task_id}, video_url: {video_url}")
                    
                    # 如果直接从JSON中提取到video_url，直接下载
                    if video_url:
                        self.log(f"从JSON中直接获取到video_url，开始下载: {video_url[:80]}...")
                        pbar.update_absolute(90)
                        video_output = self.download_and_convert_video(video_url)
                        if video_output is None:
                            error_message = "视频下载失败，但API返回了视频URL"
                            self.log(error_message)
                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message + f"\n\n视频URL: {video_url}"
                            return (SimpleVideoAdapter(""), full_text)
                        pbar.update_absolute(100)
                        response_data = {
                            "status": "success",
                            "prompt": prompt,
                            "model": model_name,
                            "size": size,
                            "seconds": seconds,
                            "video_url": video_url,
                            "full_response": full_response_text,
                            "source": "async_response_direct_json"
                        }
                        full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + json.dumps(response_data, ensure_ascii=False, indent=2)
                        return (video_output, full_text)
                    
                    # 如果已经从JSON中提取到task_id，跳过文本解析，直接轮询
                    if task_id:
                        self.log(f"从JSON中直接获取到task_id，开始轮询: {task_id}")
                        pbar.update_absolute(40)
                        # 继续执行轮询逻辑（跳过下面的文本解析部分）
                    else:
                        # 如果没有从JSON中提取到，尝试从文本中解析
                        # 匹配 task_ 前缀格式
                        task_id_match = re.search(r"ID: `(task_[a-zA-Z0-9]+)`", full_response_text)
                        if not task_id_match:
                            # 尝试匹配普通格式
                            task_id_match = re.search(r"ID: `([a-zA-Z0-9\-]+)`", full_response_text)
                        if task_id_match:
                            task_id = task_id_match.group(1)
                            self.log(f"从文本中提取到task_id: {task_id}")
                        
                        # 提取数据预览URL
                        if task_id:
                            preview_match = re.search(r"\[数据预览\]\((https://asyncdata.net/web/[^)]+)\)", full_response_text)
                            if not preview_match:
                                preview_match = re.search(r"\[数据预览\]\((https://[^)]+)\)", full_response_text)
                            if preview_match:
                                data_preview_url = preview_match.group(1)
                                self.log(f"从文本中提取到data_preview_url: {data_preview_url}")
                        
                        # 尝试从文本中提取视频URL
                        if not video_url:
                            video_match = re.search(r'\[[^\]]*\]\((https://[^\)\s]+)', full_response_text)
                            if video_match:
                                potential_url = video_match.group(1)
                                if any(domain in potential_url for domain in ['my-sora', 'gptkey', 'asyncdata', 'wcnb', '.mp4', '.webm', 'video', 'sora', 'files']):
                                    video_url = potential_url.rstrip('.,;!?')
                                    self.log(f"从文本中提取到视频URL: {video_url[:100]}...")
                                    pbar.update_absolute(95)
                            
                            if not video_url:
                                direct_url_match = re.search(r'(https://[^\s\)\]]+(?:my-sora|gptkey|asyncdata|wcnb|files)[^\s\)\]]*)', full_response_text)
                                if direct_url_match:
                                    potential_url = direct_url_match.group(1).rstrip('.,;!?')
                                    if potential_url.startswith('https://') and len(potential_url) > 20:
                                        video_url = potential_url
                                        self.log(f"从文本中提取到视频URL (直接格式): {video_url[:100]}...")
                                        pbar.update_absolute(95)
                        
                        # 如果从文本中提取到视频URL，下载并返回
                        if video_url:
                            self.log(f"开始下载并转换视频: {video_url[:80]}...")
                            pbar.update_absolute(90)
                            video_output = self.download_and_convert_video(video_url)
                            if video_output is None:
                                error_message = "视频下载失败，但API返回了视频URL"
                                self.log(error_message)
                                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message + f"\n\n视频URL: {video_url}"
                                return (SimpleVideoAdapter(""), full_text)
                            pbar.update_absolute(100)
                            response_data = {
                                "status": "success",
                                "prompt": prompt,
                                "model": model_name,
                                "size": size,
                                "seconds": seconds,
                                "video_url": video_url,
                                "full_response": full_response_text,
                                "source": "async_response_text"
                            }
                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + json.dumps(response_data, ensure_ascii=False, indent=2)
                            return (video_output, full_text)
                
                except json.JSONDecodeError as e:
                    # JSON解析失败，尝试作为文本处理
                    self.log(f"[调试] JSON解析失败: {str(e)}")
                    self.log(f"[调试] 响应原始内容: {response_text[:500]}")
                    full_response_text = response_text
                    
                    # 尝试从文本中提取task_id
                    task_id = None
                    task_id_match = re.search(r"ID: `(task_[a-zA-Z0-9]+)`", full_response_text)
                    if not task_id_match:
                        task_id_match = re.search(r"ID: `([a-zA-Z0-9\-]+)`", full_response_text)
                    if task_id_match:
                        task_id = task_id_match.group(1)
                        self.log(f"从文本中提取到task_id: {task_id}")
                
                except Exception as e:
                    error_msg = f"异步响应解析失败: {str(e)}"
                    self.log(error_msg)
                    self.log(f"[调试] 响应原始内容: {response_text[:500]}")
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_msg
                    return (SimpleVideoAdapter(""), full_text)
            
            # 如果没有提取到task_id，返回错误
            if not task_id:
                error_message = "异步调用失败：无法从响应中提取task_id或video_url"
                self.log(error_message)
                if 'full_response_text' in locals():
                    self.log(f"完整响应内容: {full_response_text[:1000]}")
                else:
                    self.log(f"响应内容: {response_text[:1000] if 'response_text' in locals() else 'N/A'}")
                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
                if 'full_response_text' in locals():
                    full_text += f"\n\n响应内容预览: {full_response_text[:1000]}"
                return (SimpleVideoAdapter(""), full_text)
            
            # 如果提取到task_id，开始轮询
            pbar.update_absolute(40)
            
            # 异步轮询任务状态 - 使用正确的API端点
            max_attempts = 120
            attempts = 0
            video_url = None
            gif_url = None
            
            # 构建查询状态的API端点
            # 根据API文档，查询任务状态使用 GET /v1/videos/{id}，需要在Body中包含model参数
            status_url = f"{base_url}/v1/videos/{task_id}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # Body参数：根据API文档，需要包含model参数
            body_data = {
                "model": model_name
            }
            
            self.log(f"开始轮询任务状态，端点: {status_url}")
            self.log(f"轮询Body参数: {json.dumps(body_data, ensure_ascii=False)}")
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    # 根据API文档，GET请求需要包含Body参数
                    # 使用json参数会自动设置Content-Type并序列化数据
                    status_response = requests.get(
                        status_url,
                        headers=headers,
                        json=body_data,  # 添加Body参数
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        error_detail = ""
                        try:
                            error_detail = status_response.json()
                        except:
                            error_detail = status_response.text[:200]
                        self.log(f"状态查询失败，状态码: {status_response.status_code}, 错误详情: {error_detail}")
                        continue
                    
                    status_data = status_response.json()
                    self.log(f"[调试] 轮询响应: {json.dumps(status_data, ensure_ascii=False)[:500]}")
                    
                    # 从响应中提取状态信息
                    status = status_data.get("status", "")
                    progress_raw = status_data.get("progress")
                    # 处理 progress 为 None 的情况
                    progress = progress_raw if progress_raw is not None else 0
                    video_id = status_data.get("id", task_id)
                    
                    # 更新进度条
                    if progress is not None and progress > 0:
                        pbar_value = min(90, 40 + int(progress * 0.5))
                        pbar.update_absolute(pbar_value)
                        self.log(f"异步任务进度: {progress}%, 状态: {status}")
                    else:
                        progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                        pbar.update_absolute(progress_value)
                        if progress_raw is None:
                            self.log(f"轮询中... ({attempts}/{max_attempts}), 状态: {status} (进度信息暂未更新)")
                        else:
                            self.log(f"轮询中... ({attempts}/{max_attempts}), 状态: {status}")
                    
                    # 检查任务状态
                    if status == "completed":
                        # 任务完成，尝试获取视频URL
                        # 方式1: 从响应中直接获取 video_url
                        video_url = status_data.get("video_url") or status_data.get("url")
                        
                        # 方式2: 如果没有直接URL，使用下载端点
                        if not video_url:
                            download_url = f"{base_url}/v1/videos/{video_id}/content"
                            self.log(f"使用下载端点获取视频: {download_url}")
                            # 先尝试获取下载URL，实际下载在下面处理
                            video_url = download_url
                        
                        self.log(f"异步视频生成完成，状态: {status}, video_id: {video_id}")
                        break
                    elif status == "failed" or status == "error":
                        error_message = status_data.get("error", {}).get("message", "Unknown error") if isinstance(status_data.get("error"), dict) else "任务失败"
                        self.log(f"异步视频生成失败: {error_message}")
                        full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + f"异步视频生成失败: {error_message}"
                        return (SimpleVideoAdapter(""), full_text)
                    elif status in ["queued", "processing", "in_progress"]:
                        # 任务还在处理中，继续轮询
                        # "in_progress" 等同于 "processing"，表示任务正在处理
                        continue
                    else:
                        # 未知状态，记录并继续轮询（避免因未知状态导致任务失败）
                        self.log(f"未知任务状态: {status}，继续轮询...")
                        continue
                        
                except json.JSONDecodeError as e:
                    self.log(f"状态查询响应解析失败: {str(e)}")
                    continue
                except Exception as e:
                    self.log(f"异步任务状态查询错误: {str(e)}")
                    continue
            
            if not video_url:
                error_message = f"异步调用失败：在{max_attempts}次尝试后仍未获取到视频URL"
                self.log(error_message)
                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
                return (SimpleVideoAdapter(""), full_text)
            
            # 下载并转换视频
            self.log(f"开始下载并转换视频: {video_url[:80]}...")
            pbar.update_absolute(90)
            
            # 检查是否是下载端点（/v1/videos/{id}/content）
            if "/v1/videos/" in video_url and "/content" in video_url:
                # 使用下载端点直接下载视频
                try:
                    download_response = requests.get(
                        video_url,
                        headers=headers,
                        timeout=self.timeout,
                        stream=True
                    )
                    
                    if download_response.status_code != 200:
                        error_message = f"视频下载失败，状态码: {download_response.status_code}"
                        self.log(error_message)
                        full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
                        return (SimpleVideoAdapter(""), full_text)
                    
                    # 获取临时目录
                    if HAS_FOLDER_PATHS:
                        temp_dir = folder_paths.get_temp_directory()
                    else:
                        temp_dir = tempfile.gettempdir()
                    
                    # 创建临时文件
                    timestamp = int(time.time())
                    temp_filename = f"sora_video_{timestamp}.mp4"
                    temp_filepath = os.path.join(temp_dir, temp_filename)
                    
                    # 下载视频到临时文件
                    with open(temp_filepath, "wb") as f:
                        for chunk in download_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    self.log(f"✅ 视频下载完成: {temp_filepath}")
                    
                    # 转换为ComfyUI VIDEO对象
                    if HAS_VIDEO_FROM_FILE and os.path.exists(temp_filepath):
                        try:
                            video_output = VideoFromFile(temp_filepath)
                            self.log(f"✅ 视频已转换为ComfyUI VIDEO对象")
                        except Exception as e:
                            self.log(f"⚠️ 无法转换为VideoFromFile，使用SimpleVideoAdapter: {str(e)}")
                            video_output = SimpleVideoAdapter(temp_filepath)
                    else:
                        video_output = SimpleVideoAdapter(temp_filepath)
                    
                except Exception as e:
                    error_message = f"视频下载过程出错: {str(e)}"
                    self.log(error_message)
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
                    return (SimpleVideoAdapter(""), full_text)
            else:
                # 使用原有的URL下载方法
                video_output = self.download_and_convert_video(video_url)
                if video_output is None:
                    error_message = "视频下载失败，但API返回了视频URL"
                    self.log(error_message)
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message + f"\n\n视频URL: {video_url}"
                    return (SimpleVideoAdapter(""), full_text)
            
            pbar.update_absolute(100)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model_name,
                "size": size,
                "seconds": seconds,
                "video_url": video_url,
                "gif_url": gif_url,
                "full_response": full_response_text if 'full_response_text' in locals() else "",
                "source": "async_polling"
            }
            
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + json.dumps(response_data, ensure_ascii=False, indent=2)
            
            return (video_output, full_text)
                
        except Exception as e:
            error_msg = f"异步调用过程出错: {str(e)}"
            self.log(error_msg)
            import traceback
            traceback.print_exc()
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_msg
            return (SimpleVideoAdapter(""), full_text)
    
    def generate_video(self, prompt, api_key, base_url, model_name, custom_model_name, seed=-1,
                       size="720x1280", seconds="10", input_reference=None, watermark=False, private=False):
        """生成视频 - 按照API格式参数"""
        response_text = ""
        self.log_messages = []
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            actual_base_url = self.get_base_url(base_url).rstrip('/')
            
            # 处理模型名称
            if model_name == "自定义输入 (Custom Input)":
                actual_model_name = custom_model_name.strip() if custom_model_name and custom_model_name.strip() else "sora-2"
                if not actual_model_name or actual_model_name == "":
                    actual_model_name = "sora-2"
                    self.log("警告: 自定义模型名称为空，使用默认值 sora-2")
            else:
                actual_model_name = model_name if model_name else "sora-2"
            
            # 验证 seconds 参数
            if actual_model_name == "sora-2":
                if seconds == "25":
                    error_message = "The sora-2 model does not support 25 second videos. Please use sora-2-pro for 25 second videos."
                    self.log(error_message)
                    full_text = "## 错误\n" + error_message
                    return (SimpleVideoAdapter(""), full_text)
            
            self.log(f"最终配置 - URL: {actual_base_url}, 模型: {actual_model_name}, 尺寸: {size}, 时长: {seconds}秒, 水印: {watermark}, 私有: {private}")
            
            # 验证 prompt
            if not prompt or (isinstance(prompt, str) and prompt.strip() == ""):
                error_message = "错误: prompt 不能为空，请输入视频生成的提示词。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message
                return (SimpleVideoAdapter(""), full_text)
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message
                return (SimpleVideoAdapter(""), full_text)
            
            # 初始化进度条
            if HAS_COMFY_UTILS:
                pbar = comfy.utils.ProgressBar(100)
            else:
                pbar = SimpleProgressBar(100)
            pbar.update_absolute(10)
            
            # 直接使用异步调用
            self.log("使用异步调用模式生成视频")
            return self._try_async_call(actual_api_key, actual_base_url, actual_model_name, 
                                       prompt, input_reference, seed, size, seconds, watermark, private, pbar)
        
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"视频生成错误: {str(e)}")
            traceback.print_exc()
            
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
            return (SimpleVideoAdapter(""), full_text)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Google-Gemini": GeminiImageGenerator,
    "Gemini-Image-To-Prompt": GeminiImageToPrompt,
    "sora-Prompt-To-video": soraPromptToVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google-Gemini": "双子座3.0图像",
    "Gemini-Image-To-Prompt": "双子座图片提示",
    "sora-Prompt-To-Video": "sora提示转视频"
} 

