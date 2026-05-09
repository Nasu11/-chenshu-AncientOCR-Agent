# config.py
import os
from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    # 模型路径（请根据实际路径调整）
    BASE_MODEL: str = "models/Qwen2-VL-2B-Instruct"
    LORA_OCR: str = "qwen2vl_ocr_full/final_model"
    LORA_GROUND: str = "qwen2vl_grounding/final_model"
    
    # 图像根目录
    IMG_ROOTS: list = field(default_factory=lambda: [
        "中华书局_陈书/陈书.第1册.卷一至卷一六",
        "中华书局_陈书/陈书.第2册.卷一七至卷三六"
    ])
    
    # 推理参数
    DEVICE: str = "cuda:0"
    MAX_TOKENS_OCR: int = 2048
    MAX_TOKENS_GROUND: int = 256
    DTYPE: str = "float16"

cfg = AgentConfig()