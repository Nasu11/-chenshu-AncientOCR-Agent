# graph/workflow.py
import json, re
from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END
from config import cfg
from tools.registry import TOOL_MAP

class AgentState(TypedDict):
    user_query: str
    image_path: str
    messages: Annotated[List[dict], operator.add]
    tool_calls: List[dict]
    tool_results: dict
    next_step: str
    final_output: str

def router_node(state: AgentState) -> AgentState:
    """确定性路由扩展：支持单工具、多步链式、基座兜底"""
    q = state["user_query"].lower()
    chain_steps = []
    
    # 1. 核心任务识别（按语义顺序追加至链条）
    if any(k in q for k in ["全文", "整页", "所有文字", "识别全", "full", "ocr"]):
        chain_steps.append({"tool": "full_text_ocr_tool", "args": {"image_path": state["image_path"]}})
    elif any(k in q for k in ["坐标", "位置", "在哪", "找出", "定位", "ground"]):
        match = re.search(r'["\']([^"\']+)["\']|找出\s*([^\s，。]+)|定位\s*([^\s，。]+)', state["user_query"])
        target = (match.group(1) or match.group(2) or match.group(3) or "文本").strip()
        chain_steps.append({"tool": "visual_grounding_tool", "args": {"image_path": state["image_path"], "query": target}})
    elif any(k in q for k in ["版面", "结构", "layout"]):
        chain_steps.append({"tool": "layout_analysis_tool", "args": {"image_path": state["image_path"]}})
    
    # 2. 后置链式任务识别（自动追加至链条末尾）
    if any(k in q for k in ["纠错", "修正", "correct", "改", "替换"]):
        chain_steps.append({"tool": "text_correction_tool", "args": {"raw_text": "${prev_output}", "context": state["user_query"]}})
    if any(k in q for k in ["导出", "转换", "export", "csv", "json", "xml", "格式"]):
        fmt = "csv" if "csv" in q else ("xml" if "xml" in q else "json")
        chain_steps.append({"tool": "format_export_tool", "args": {"structured_data": "${prev_output}", "target_format": fmt}})
        
    # 3. 规则未匹配 → 基座模型兜底
    if not chain_steps:
        chain_steps.append({"tool": "base_model_chat_tool", "args": {"query": state["user_query"], "image_path": state["image_path"]}})
        
    state["tool_calls"] = chain_steps
    state["next_step"] = "execute_chain"
    return state

def tool_executor_node(state: AgentState) -> AgentState:
    """多步链式执行节点：支持占位符解析与中间结果透传"""
    try:
        chain = state["tool_calls"]
        prev_raw_output = ""
        results = {}
        
        for i, call in enumerate(chain):
            tool_name = call["tool"]
            args = call["args"].copy()
            
            # 解析占位符：${prev_output} 替换为上一工具原始输出
            for k, v in args.items():
                if isinstance(v, str) and "${prev_output}" in v:
                    args[k] = prev_raw_output
            
            tool_func = TOOL_MAP.get(tool_name)
            if not tool_func:
                raise ValueError(f"未注册工具: {tool_name}")
            
            print(f"⚙️ [Chain {i+1}/{len(chain)}] 执行工具: {tool_name}")
            result = tool_func(**args)
            result_json = json.loads(result)
            
            # 提取原始文本供下游使用（兼容 OCR/Correction/Base 的输出结构）
            prev_raw_output = (
                result_json.get("raw_output") or 
                result_json.get("response") or 
                json.dumps(result_json, ensure_ascii=False)
            )
            results[tool_name] = result_json
            
        state["tool_results"] = results
        state["next_step"] = "post_process"
        
    except Exception as e:
        print(f"🔴 工具执行异常: {e}")
        state["tool_results"] = {"error": str(e)}
        state["next_step"] = "final_answer"
    return state

def post_process_node(state: AgentState) -> AgentState:
    """后处理与多步决策节点"""
    # 示例：若结果为 layout，自动触发 format_export
    if "layout_analysis_tool" in state["tool_results"]:
        state["tool_calls"] = [{"tool": "format_export_tool", "args": {"structured_data": json.dumps(state["tool_results"]["layout_analysis_tool"]), "target_format": "json"}}]
        state["next_step"] = "execute_tool"
    else:
        state["next_step"] = "final_answer"
    return state

def final_answer_node(state: AgentState) -> AgentState:
    """终态组装：优先返回最后一步工具结果"""
    if "error" in state.get("tool_results", {}):
        state["final_output"] = f"⚠️ 执行失败: {state['tool_results']['error']}"
    else:
        # 链式执行时，返回最终工具的结构化输出
        last_tool = state["tool_calls"][-1]["tool"]
        state["final_output"] = json.dumps(state["tool_results"].get(last_tool, {}), ensure_ascii=False, indent=2)
        
    state["messages"].append({"role": "assistant", "content": state["final_output"]})
    return state

def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("execute_tool", tool_executor_node)
    workflow.add_node("post_process", post_process_node)
    workflow.add_node("final_answer", final_answer_node)
    
    workflow.set_entry_point("router")
    # ✅ 新代码：支持单步/链式执行
    workflow.add_conditional_edges("router", lambda s: s["next_step"], {
        "execute_tool": "execute_tool",      # 单步执行（向后兼容）
        "execute_chain": "execute_tool"      # 链式执行（复用同一节点）
    })
    workflow.add_conditional_edges("execute_tool", lambda s: s["next_step"], {
        "post_process": "post_process",
        "execute_tool": "execute_tool",      # 支持链式循环调用
        "final_answer": "final_answer"
    })
    workflow.add_edge("post_process", "final_answer")
    workflow.add_edge("final_answer", END)
    
    return workflow.compile()