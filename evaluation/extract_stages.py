#!/usr/bin/env python3
"""
从 history.jsonl 中提取各阶段的中间过程输出，保存为独立的 markdown 文件。

功能：
- 解析 JSONL 中的每个事件
- 提取 GENERATOR、REVISER、VERIFIER 三个阶段的 content、verification 字段
- 保存为单独的 .md 文件，保留真实的换行符和格式

使用方法：
    python extract_stages.py <jsonl_file> [output_dir]

示例：
    # 提取到同目录
    python extract_stages.py runs/imo-bench-algebra-001_20260502_151156/history.jsonl
    
    # 提取到指定目录
    python extract_stages.py runs/imo-bench-algebra-001_20260502_151156/history.jsonl ./extracted
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional


def extract_stages(jsonl_path: str, output_dir: Optional[str] = None) -> int:
    """
    从 JSONL 或 JSON 数组文件提取所有阶段的输出。
    
    支持两种格式：
    1. JSONL：每行一个 JSON 对象
    2. JSON 数组：整个文件是一个 JSON 数组
    
    Args:
        jsonl_path: history.jsonl 文件路径
        output_dir: 输出目录（默认：history.jsonl 所在目录）
        
    Returns:
        提取的文件数量
    """
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        print(f"❌ 错误：文件不存在 {jsonl_path}")
        return 0
    
    if output_dir is None:
        output_dir = jsonl_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 读取 JSONL/JSON 文件：{jsonl_path}")
    
    # 读取文件并解析
    events = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 判断是 JSON 数组还是 JSONL
    if content.startswith('['):
        # JSON 数组格式
        print("  📋 检测到 JSON 数组格式")
        try:
            events = json.loads(content)
            if not isinstance(events, list):
                print("❌ 错误：JSON 不是数组格式")
                return 0
        except json.JSONDecodeError as e:
            print(f"❌ JSON 数组解析失败：{e}")
            return 0
    else:
        # JSONL 格式
        print("  📋 检测到 JSONL 格式")
        lines = content.split('\n')
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  警告：第 {line_no} 行 JSON 解析失败 - {e}")
                continue
    
    print(f"✓ 成功读取 {len(events)} 个事件\n")
    
    # 提取各阶段的 content
    stage_mapping = {
        "GENERATOR": "generator",
        "REVISER": "reviser",
        "VERIFIER": "verifier"
    }
    
    saved_files = []
    
    for event in events:
        node = event.get("node")
        if node in stage_mapping:
            # 不同阶段的内容可能存储在不同的字段
            if node == "VERIFIER":
                # VERIFIER 的内容存储在 verification 字段
                content = event.get("verification", "")
            else:
                # GENERATOR 和 REVISER 的内容存储在 content 字段
                content = event.get("content", "")
            
            if not content:
                print(f"⚠️  跳过：{node} 阶段内容为空")
                continue
            
            stage_name = stage_mapping[node]
            turn_id = event.get("turn_id", 0)
            
            # 生成文件名
            filename = f"{turn_id}_{stage_name}.md"
            filepath = output_dir / filename
            
            # 保存内容
            # 关键：content 是字符串，包含真实的换行符（从 JSON 反序列化后）
            with open(filepath, 'w', encoding='utf-8') as f:
                r"""规范化 lemma 内容中的展示数学格式，把文本中的 LaTeX 显示数学公式标记 \[ ... \] 转换成 Markdown/数学渲染器更通用的 $$ ... $$ 格式。"""
                normalized = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.S)
                f.write(normalized)
            
            saved_files.append(filepath)
            
            # 显示文件信息
            content_preview = normalized[:80].replace('\n', ' ').strip()
            if len(normalized) > 80:
                content_preview += "..."
            print(f"✅ 已保存：{filename}")
            print(f"   📄 大小：{len(normalized)} 字符")
            print(f"   📝 内容预览：{content_preview}\n")
    
    if not saved_files:
        print("❌ 没有找到任何可提取的阶段输出")
        return 0
    
    print("=" * 70)
    print(f"✨ 成功提取 {len(saved_files)} 个文件")
    print(f"📁 输出目录：{output_dir.resolve()}")
    print("=" * 70)
    
    # 显示文件列表
    print("\n提取的文件：")
    for filepath in sorted(saved_files):
        print(f"  • {filepath.name}")
    
    return len(saved_files)


def main(argv: Optional[list] = None) -> int:
    """主入口"""
    if argv is None:
        argv = sys.argv[1:]
    
    if not argv or argv[0] in ['-h', '--help']:
        print(__doc__)
        return 0 if argv and argv[0] in ['-h', '--help'] else 1
    
    jsonl_path = argv[0]
    output_dir = argv[1] if len(argv) > 1 else None
    
    try:
        count = extract_stages(jsonl_path, output_dir)
        return 0 if count > 0 else 1
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
