#!/usr/bin/env python3
"""从 history.jsonl 中提取各阶段的中间过程输出，保存为独立的 markdown 文件。

功能：
- 解析 JSONL 或 JSON 数组格式的事件日志
- 提取三个阶段的输出：
  * GENERATOR: content（解答）和 reasoning_content（推理过程）[新增]
  * REVISER: content（修正内容）
  * VERIFIER: verification（验证结果）
- 保存为单独的 .md 文件，保留真实的换行符和格式

关键特性（更新）：
- 现在从 GENERATOR 阶段提取两个文件：
  * {turn_id}_generator.md - 模型生成的解答
  * {turn_id}_generator_reasoning.md - 模型的中间推理过程（用于分析思维过程）

使用方法：
    python extract_stages.py <jsonl_file> [output_dir]
    python -m evaluation.extract_stages <jsonl_file> [output_dir]

示例：
    # 提取到同目录
    python -m evaluation.extract_stages runs/imo-bench-algebra-001_20260502_151156/history.jsonl
    
    # 提取到指定目录
    python -m evaluation.extract_stages runs/imo-bench-algebra-001_20260502_151156/history.jsonl ./extracted
    
    # 查看提取结果
    ls -la runs/imo-bench-algebra-001_20260502_151156/extracted/
    cat runs/imo-bench-algebra-001_20260502_151156/extracted/0_generator_reasoning.md

集成说明：
- run_and_analyze_selected.py 会为所有非 SUCCESS 题目自动调用本脚本
- 自动调用命令：python -m evaluation.extract_stages history.jsonl extracted/
- 提取失败不会中断主流程
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional


def normalize_latex_display_math(text: str) -> str:
    r"""规范化文本中的 LaTeX 显示数学公式格式。

    将 LaTeX 的 \[ ... \] 、\( ... \) 显示数学标记转换为 Markdown 通用的 $$ ... $$ 格式，
    确保在各类 Markdown 渲染器（如 Jupyter、GitHub、VS Code 等）中正确显示。

    处理方式：
    - 使用非贪婪匹配 (.*?) 配合 re.S 标志，支持跨行匹配
    - 保留公式内部的所有原始内容（包括换行符和缩进）
    - 仅替换标记符本身，不修改公式内容

    Args:
        text: 包含 LaTeX 显示数学标记的原始文本

    Returns:
        规范化后的文本，\[ ... \] 被替换为 $$ ... $$
    """
    return re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.S)


def extract_stages(jsonl_path: str, output_dir: Optional[str] = None) -> int:
    """从 JSONL 或 JSON 数组文件提取所有阶段的输出。
    
    支持两种格式：
    1. JSONL：每行一个 JSON 对象
    2. JSON 数组：整个文件是一个 JSON 数组
    
    提取内容：
    - GENERATOR 阶段:
      * content → {turn_id}_generator.md (模型的解答)
      * reasoning_content → {turn_id}_generator_reasoning.md (模型的推理过程) [新增]
    - REVISER 阶段:
      * content → {turn_id}_reviser.md (修正内容)
    - VERIFIER 阶段:
      * verification → {turn_id}_verifier.md (验证结果)
    
    输出目录结构示例：
    ```
    runs/PB-Basic-001_20260519_093502/extracted/
    ├── 0_generator.md              # 第1轮生成的解答
    ├── 0_generator_reasoning.md    # 第1轮的推理过程 (NEW)
    ├── 1_generator.md              # 第2轮生成的解答
    ├── 1_generator_reasoning.md    # 第2轮的推理过程 (NEW)
    ├── 0_reviser.md                # 第1轮修正
    └── 0_verifier.md               # 第1轮验证
    ```
    
    使用示例：
    ```python
    # 提取到当前目录的 extracted 子目录
    extract_stages("runs/PB-Basic-001_20260519_093502/history.jsonl")
    
    # 指定输出目录
    extract_stages("runs/PB-Basic-001_20260519_093502/history.jsonl", "./extracted")
    ```
    
    Args:
        jsonl_path: history.jsonl 文件路径
        output_dir: 输出目录（默认：history.jsonl 所在目录）
        
    Returns:
        提取的文件数量（包括 reasoning_content 文件）
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
    
    # 提取各阶段的 content 和 reasoning_content
    # 关键更新：GENERATOR 现在提取两个文件
    # 1. {turn_id}_generator.md - 模型生成的解答
    # 2. {turn_id}_generator_reasoning.md - 模型的中间推理过程 (NEW)
    # 这两个文件用于分析模型的思考和解题过程
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
                reasoning_content = None
            else:
                # GENERATOR 和 REVISER 的内容存储在 content 字段
                content = event.get("content", "")
                # GENERATOR 也包含 reasoning_content 字段
                reasoning_content = event.get("reasoning_content", None)
                if isinstance(reasoning_content, list):
                    reasoning_content = "\n\n".join(
                        f"### 第 {i+1} 轮推理\n{item}"
                        for i, item in enumerate(reasoning_content)
                        if item
                    )
            
            if not content and not reasoning_content:
                print(f"⚠️  跳过：{node} 阶段内容为空")
                continue
            
            stage_name = stage_mapping[node]
            turn_id = event.get("turn_id", 0)
            
            # 首先保存主要内容（content 或 verification）
            if content:
                filename = f"{turn_id}_{stage_name}.md"
                filepath = output_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    # 复用规范化函数，将 LaTeX 显示数学标记转为 Markdown 通用格式
                    normalized = normalize_latex_display_math(content)
                    f.write(normalized)
                
                saved_files.append(filepath)
                
                content_preview = normalized[:80].replace('\n', ' ').strip()
                if len(normalized) > 80:
                    content_preview += "..."
                print(f"✅ 已保存：{filename}")
                print(f"   📄 大小：{len(normalized)} 字符")
                print(f"   📝 内容预览：{content_preview}\n")
            
            # 如果存在 reasoning_content（仅 GENERATOR 有），单独保存
            if reasoning_content and node == "GENERATOR":
                reasoning_filename = f"{turn_id}_{stage_name}_reasoning.md"
                reasoning_filepath = output_dir / reasoning_filename
                
                with open(reasoning_filepath, 'w', encoding='utf-8') as f:
                    # 复用规范化函数，统一处理推理内容中的数学公式格式
                    normalized = normalize_latex_display_math(reasoning_content)
                    f.write(normalized)
                
                saved_files.append(reasoning_filepath)
                
                print(f"✅ 已保存：{reasoning_filename}")
                print(f"   📄 大小：{len(normalized)} 字符\n")
    
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
    """主入口 - 提取 history.jsonl 中的所有阶段输出。
    
    使用示例：
    ```bash
    # 基本用法：提取到同级 extracted 目录
    python -m evaluation.extract_stages runs/PB-Basic-001_20260519_093502/history.jsonl
    
    # 指定输出目录
    python -m evaluation.extract_stages runs/PB-Basic-001_20260519_093502/history.jsonl ./extracted
    
    # 查看生成的文件
    ls -la runs/PB-Basic-001_20260519_093502/extracted/
    cat runs/PB-Basic-001_20260519_093502/extracted/0_generator_reasoning.md
    ```
    
    自动调用说明：
    - run_and_analyze_selected.py 在非 SUCCESS 题目完成后会自动调用本脚本
    - 命令：python -m evaluation.extract_stages {history.jsonl} {run_dir/extracted}
    
    参数说明：
    - argv[0]: history.jsonl 文件路径（必需）
    - argv[1]: 输出目录（可选，默认：history.jsonl 同级目录）
    
    输出文件类型（新增 reasoning_content）：
    - {turn_id}_generator.md: Generator 的解答
    - {turn_id}_generator_reasoning.md: Generator 的推理过程 (NEW!)
    - {turn_id}_reviser.md: Reviser 的修正
    - {turn_id}_verifier.md: Verifier 的验证
    """
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
