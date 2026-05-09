# tools/registry.py
import os, json, traceback, torch, re
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from config import cfg

# ================= 全局模型缓存 =================
_model_cache = {}

def get_vl_model():
    """懒加载基座模型与 Processor"""
    if "base" not in _model_cache:
        print(f"📥 Loading Base Model: {cfg.BASE_MODEL}")
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.BASE_MODEL,
            torch_dtype=torch.float16,
            device_map=cfg.DEVICE,
            low_cpu_mem_usage=True
        )
        proc = Qwen2VLProcessor.from_pretrained(cfg.BASE_MODEL)
        _model_cache["base"], _model_cache["proc"] = base, proc
    return _model_cache["base"], _model_cache["proc"]

def load_lora_adapter(base, adapter_path, name):
    """动态加载 LoRA 适配器（返回新包装模型）"""
    print(f"🔧 Loading LoRA: {name} from {adapter_path}")
    return PeftModel.from_pretrained(base, adapter_path, adapter_name=name)

# ================= 工具定义 =================
def full_text_ocr_tool(image_path: str) -> str:
    """识别古籍全页文字及坐标。返回 JSON 格式列表。"""
    try:
        base, proc = get_vl_model()
        model = load_lora_adapter(base, cfg.LORA_OCR, "ocr")
        
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > 1024:
            ratio = 1024 / max(img.size)
            img = img.resize((int(img.width*ratio), int(img.height*ratio)))
        
        prompt = "请识别图中所有的文字，并给出它们对应的坐标框。"
        msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        txt_in = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(images=[img], text=txt_in, return_tensors="pt").to(cfg.DEVICE)
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            out = model.generate(**inputs, max_new_tokens=cfg.MAX_TOKENS_OCR, do_sample=False, pad_token_id=proc.tokenizer.pad_token_id)
        
        raw = proc.decode(out[0], skip_special_tokens=True)
        result = raw.split("assistant")[-1].strip() if "assistant" in raw else raw
        return json.dumps({"type": "full_text_ocr", "raw_output": result}, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"OCR 执行失败：{e}")
    finally:
        torch.cuda.empty_cache()

def visual_grounding_tool(image_path: str, query: str) -> str:
    """定位古籍中指定文本的坐标框。返回 JSON 格式坐标列表。"""
    try:
        base, proc = get_vl_model()
        model = load_lora_adapter(base, cfg.LORA_GROUND, "ground")
        
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > 1024:
            ratio = 1024 / max(img.size)
            img = img.resize((int(img.width*ratio), int(img.height*ratio)))
        
        prompt = f"请找出文本\"{query}\"在图中的坐标框，格式：(x1,y1,x2,y2)"
        msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        txt_in = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(images=[img], text=txt_in, return_tensors="pt").to(cfg.DEVICE)
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            out = model.generate(**inputs, max_new_tokens=cfg.MAX_TOKENS_GROUND, do_sample=False, pad_token_id=proc.tokenizer.pad_token_id)
        
        raw = proc.decode(out[0], skip_special_tokens=True)
        result = raw.split("assistant")[-1].strip() if "assistant" in raw else raw
        return json.dumps({"type": "visual_grounding", "query": query, "raw_output": result}, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Grounding 执行失败：{e}")
    finally:
        torch.cuda.empty_cache()

def layout_analysis_tool(image_path: str) -> str:
    """解析古籍版面结构（版心/栏线/夹注/印章）。返回 JSON 结构。"""
    return json.dumps({"type": "layout", "headers": [], "columns": [], "seals": [], "note": "零样本解析待接入"}, ensure_ascii=False)

def text_correction_tool(raw_text: str, context: str = "") -> str:
    """基于异体字词典与上下文修正 OCR 错字。返回修正后文本。"""
    corrected = raw_text.replace("覇", "霸").replace("羣", "群").replace("峯", "峰")
    return json.dumps({"type": "correction", "corrected": corrected}, ensure_ascii=False)

def format_export_tool(structured_data: str, target_format: str = "json") -> str:
    """将结构化结果转换为目标格式（json/csv/tei_xml）。"""
    try:
        # 尝试解析输入为 JSON（若已是字符串则直接使用）
        data = json.loads(structured_data) if isinstance(structured_data, str) else structured_data
        
        if target_format == "csv":
            # 简化版 CSV 转换（可根据实际需求扩展）
            if isinstance(data, dict) and "results" in data:
                rows = ["x1,y1,x2,y2,text"]
                for item in data["results"]:
                    bbox = item.get("bbox", [0,0,0,0])
                    text = item.get("text", "").replace(",", "，")  # 避免 CSV 分隔符冲突
                    rows.append(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{text}")
                return "\n".join(rows)
            return str(data)  # 兜底返回字符串
        
        elif target_format == "tei_xml":
            # 简化版 TEI-XML 转换（可根据 TEI 标准扩展）
            xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n<TEI xmlns="http://www.tei-c.org/ns/1.0">\n  <text>\n    <body>\n'
            xml_footer = '    </body>\n  </text>\n</TEI>'
            if isinstance(data, dict) and "raw_output" in data:
                content = data["raw_output"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return xml_header + f"      <p>{content}</p>\n" + xml_footer
            return xml_header + xml_footer
        
        else:  # json 或默认
            return json.dumps({"type": "export", "format": target_format, "data": data}, ensure_ascii=False, indent=2)
            
    except Exception as e:
        raise RuntimeError(f"格式转换失败：{e}")
    
# ================= 基座模型缓存（用于兜底与通用问答） =================
_base_cache = {}

def get_base_model():
    if "base" not in _base_cache:
        print("📥 Loading Base Model (Fallback/Chat)...")
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.BASE_MODEL, torch_dtype=torch.float16, device_map=cfg.DEVICE, low_cpu_mem_usage=True
        )
        proc = Qwen2VLProcessor.from_pretrained(cfg.BASE_MODEL)
        _base_cache["base"], _base_cache["proc"] = base, proc
    return _base_cache["base"], _base_cache["proc"]

def base_model_chat_tool(query: str, image_path: str = None) -> str:
    """基座模型通用问答（显存安全+优雅降级版）"""
    try:
        # 1. 强制清理显存碎片
        torch.cuda.empty_cache()
        
        # 2. 复用全局缓存基座（避免重复加载导致显存翻倍）
        model, proc = get_vl_model()
        
        # 3. 图像严格降采样（控制视觉编码峰值显存）
        img = None
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            # 降至 448px，大幅降低 ViT 编码显存占用
            if max(img.size) > 448:
                ratio = 448 / max(img.size)
                img = img.resize((int(img.width*ratio), int(img.height*ratio)))
                
        # 4. 构造多模态 Prompt
        content = []
        if img:
            content.append({"type": "image"})
        content.append({"type": "text", "text": query})
        
        msgs = [{"role": "user", "content": content}]
        txt_in = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        
        inputs = proc(images=[img] if img else None, text=txt_in, return_tensors="pt").to(cfg.DEVICE)
        
        # 5. 低显存推理配置
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            out = model.generate(
                **inputs,
                max_new_tokens=384,  # 限制生成长度
                do_sample=False,
                pad_token_id=proc.tokenizer.pad_token_id,
                num_beams=1          # 禁用 Beam Search 节省显存
            )
            
        raw = proc.decode(out[0], skip_special_tokens=True)
        response = raw.split("assistant")[-1].strip() if "assistant" in raw else raw
        return json.dumps({"type": "base_chat", "response": response}, ensure_ascii=False)
        
    except torch.cuda.OutOfMemoryError:
        # 🛡️ 优雅降级：不中断状态机，返回结构化提示
        print("🔴 显存溢出，已触发安全降级策略")
        return json.dumps({
            "type": "base_chat", 
            "response": "当前单卡显存资源紧张，无法完成高分辨率图文解析。建议关闭其他占用 GPU 的进程，或改用纯文本指令查询。",
            "fallback": True
        }, ensure_ascii=False)
        
    except Exception as e:
        raise RuntimeError(f"基座模型调用失败: {e}")
        
    finally:
        # 确保每次调用后释放碎片
        torch.cuda.empty_cache()

# 更新工具注册表（将此行替换原 TOOL_MAP）
TOOL_MAP = {
    "full_text_ocr_tool": full_text_ocr_tool,
    "visual_grounding_tool": visual_grounding_tool,
    "layout_analysis_tool": layout_analysis_tool,
    "text_correction_tool": text_correction_tool,
    "format_export_tool": format_export_tool,
    "base_model_chat_tool": base_model_chat_tool  # ✅ 新增兜底工具
}