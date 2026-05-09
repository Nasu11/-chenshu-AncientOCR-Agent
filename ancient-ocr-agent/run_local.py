# run_local.py
import os, sys, json, argparse
from graph.workflow import build_agent_graph
from config import cfg

def resolve_image_path(rel_path: str) -> str:
    if os.path.isabs(rel_path): return rel_path
    for root in cfg.IMG_ROOTS:
        full = os.path.join(root, rel_path)
        if os.path.exists(full): return full
    return rel_path

def main():
    parser = argparse.ArgumentParser(description="古籍 OCR Agent (本地模式)")
    parser.add_argument("--image", type=str, required=True, help="图像路径")
    parser.add_argument("--query", type=str, default="识别这一页的所有文字", help="用户指令")
    parser.add_argument("--output", type=str, default="outputs/result.json", help="结果保存路径")
    args = parser.parse_args()
    
    print("🚀 初始化 LangGraph Agent...")
    agent = build_agent_graph()
    
    initial_state = {
        "user_query": args.query,
        "image_path": resolve_image_path(args.image),
        "messages": [],
        "tool_calls": [],
        "tool_results": {},
        "next_step": "",
        "final_output": ""
    }
    
    print(f"📝 任务: {args.query}")
    print(f"🖼️  图像: {initial_state['image_path']}")
    
    final_state = agent.invoke(initial_state)
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
        
    print("\n✅ 执行完成")
    print(f"🔍 路由: {final_state.get('tool_calls')}")
    print(f"📄 输出预览: {final_state['final_output'][:300]}...")
    print(f"💾 结果已保存: {args.output}")

if __name__ == "__main__":
    main()