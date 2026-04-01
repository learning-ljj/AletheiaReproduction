"""Low-coupling Markdown -> LaTeX -> PDF pipeline inspired by ResearchClaw.

The design goal is not to re-create every template from the original repo.
Instead, this module keeps the reusable core:
1. Template selection.
2. Markdown-to-LaTeX conversion.
3. Local style-file copying.
4. pdflatex / bibtex compilation with error capture.

中文说明：
- 本模块提供把简化的 Markdown 论文草稿转换为可编译的 LaTeX 的工具链，
    包括模板选择、行内 markdown 转 LaTeX、代码/图片处理，以及可选的 pdflatex 编译并收集错误信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import re
import shutil
import subprocess

from docs.analysis.migration_core.shared_models import CompileResult, ConferenceTemplate


# Keep template configuration local and small so migration stays simple.
# 模板配置保持精简并内置少量轻量模板，方便直接迁移使用或快速定制。
TEMPLATES: dict[str, ConferenceTemplate] = {
    "generic": ConferenceTemplate(
        name="generic",
        document_class="article",
        class_options="11pt",
        bibliography_style="plainnat",
        use_natbib=True,
        extra_preamble=(
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{amsmath,amssymb}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{natbib}\n"
        ),
    ),
    "neurips_2025": ConferenceTemplate(
        name="neurips_2025",
        document_class="article",
        class_options="final",
        bibliography_style="plainnat",
        use_natbib=True,
        extra_preamble=(
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{url}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{amsfonts}\n"
            "\\usepackage{nicefrac}\n"
            "\\usepackage{microtype}\n"
            "\\usepackage{xcolor}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{natbib}\n"
        ),
    ),
}


def get_template(name: str) -> ConferenceTemplate:
    """Return a known template or fail early with a clear error."""
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template: {name}")
    return TEMPLATES[name]

# 获取内置模板的便捷函数，遇到未知模板会抛出异常以便快速发现配置错误。


def sanitize_latex(text: str) -> str:
    """Escape characters that would otherwise break LaTeX parsing."""
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    output = text
    for source, target in replacements.items():
        output = output.replace(source, target)
    return output

# 对可能破坏 LaTeX 语法的字符做转义，供行内文本与标题等使用。


def convert_inline_markdown(text: str) -> str:
    """Convert a small subset of inline markdown into LaTeX commands."""
    text = re.sub(r"`([^`]+)`", lambda match: "\\texttt{" + sanitize_latex(match.group(1)) + "}", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", text)
    text = re.sub(r"\[([A-Za-z0-9:_-]+)\]", r"\\cite{\1}", text)
    return text

# 将常见的行内 Markdown 标记（code / bold / italic / cite key）转换为 LaTeX 对应命令。


def extract_title(markdown_text: str) -> str:
    """Use the first H1 as paper title, or fall back to a generic title."""
    match = re.search(r"^#\s+(.+)$", markdown_text, flags=re.M)
    return match.group(1).strip() if match else "Untitled Paper"

# 提取标题：以第一个 H1 作为论文标题，找不到则返回 "Untitled Paper"。


def extract_abstract(markdown_text: str) -> str:
    """Treat the first block after an 'Abstract' heading as the abstract."""
    match = re.search(r"^##\s+Abstract\s*$\n+(.+?)(?=\n##\s+|\Z)", markdown_text, flags=re.M | re.S | re.I)
    if not match:
        return ""
    abstract = match.group(1).strip()
    return convert_inline_markdown(sanitize_latex(re.sub(r"\n+", " ", abstract)))

# 抽取摘要：寻找第一个 "## Abstract" 段落并作为摘要，做行内标记转换与转义。


def split_blocks(markdown_text: str) -> list[str]:
    """Split markdown into paragraphs and fenced code blocks."""
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for line in markdown_text.splitlines():
        if line.startswith("```"):
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            in_code = not in_code
            current.append(line)
            continue
        if not in_code and not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]

# 将 Markdown 文本切分为段落（或围栏代码块），便于逐块渲染为 LaTeX。


def render_block(block: str) -> str:
    """Convert one markdown block to LaTeX."""
    if block.startswith("```") and block.endswith("```"):
        # 围栏代码块直接映射到 verbatim，最稳，也最容易迁移。
        lines = block.splitlines()
        code_lines = lines[1:-1]
        code = "\n".join(code_lines)
        return "\\begin{verbatim}\n" + code + "\n\\end{verbatim}"
    if block.startswith("### "):
        return "\\subsection{" + sanitize_latex(block[4:].strip()) + "}"
    if block.startswith("## "):
        return "\\section{" + sanitize_latex(block[3:].strip()) + "}"
    if block.startswith("# "):
        return ""
    image_match = re.match(r"!\[(.*?)\]\((.*?)\)", block.strip())
    if image_match:
        # 图片块转成最基础的 figure 环境，不绑定复杂模板能力。
        caption = sanitize_latex(image_match.group(1).strip())
        path = image_match.group(2).strip()
        return (
            "\\begin{figure}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.9\\linewidth]{{{path}}}\n"
            f"\\caption{{{caption}}}\n"
            "\\end{figure}"
        )
    paragraph = sanitize_latex(block)
    # 段落级文本先做转义，再处理行内 markdown 标记。
    paragraph = convert_inline_markdown(paragraph)
    paragraph = paragraph.replace("\n", " ")
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    return paragraph + "\n"

# 将单个块（段落 / 标题 / 图片 / 代码块）渲染为 LaTeX 对应结构。


def build_body(markdown_text: str) -> str:
    """Convert the whole markdown body except title and abstract."""
    blocks = split_blocks(markdown_text)
    rendered: list[str] = []
    skip_next_abstract_body = False
    for block in blocks:
        if re.match(r"^#\s+", block):
            continue
        if re.match(r"^##\s+Abstract\s*$", block, flags=re.I):
            skip_next_abstract_body = True
            continue
        if skip_next_abstract_body:
            skip_next_abstract_body = False
            continue
        rendered_block = render_block(block)
        if rendered_block:
            rendered.append(rendered_block)
    return "\n\n".join(rendered)

# 将全文的主体部分逐块渲染为 LaTeX（跳过标题与摘要部分），组合为最终正文文本。


def render_preamble(template: ConferenceTemplate, title: str, authors: Iterable[str]) -> str:
    """Render the top of paper.tex from template metadata."""
    class_options = f"[{template.class_options}]" if template.class_options else ""
    author_line = " \\\\ ".join(sanitize_latex(author) for author in authors) or "Anonymous Authors"
    return (
        f"\\documentclass{class_options}{{{template.document_class}}}\n"
        f"{template.extra_preamble}\n"
        f"\\title{{{sanitize_latex(title)}}}\n"
        f"\\author{{{author_line}}}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
    )

# 渲染 LaTeX 文档的前导部分：文档类、preamble、标题与作者行。


def render_footer(template: ConferenceTemplate, bib_file: str) -> str:
    """Render the bibliography footer and document end marker."""
    footer = []
    if template.use_natbib:
        footer.append(f"\\bibliographystyle{{{template.bibliography_style}}}")
        footer.append(f"\\bibliography{{{bib_file}}}")
    footer.append("\\end{document}")
    return "\n".join(footer) + "\n"

# 渲染文档尾部，包含参考文献样式和 \end{document}。


def markdown_to_latex(
    markdown_text: str,
    template: ConferenceTemplate,
    *,
    title: str | None = None,
    authors: list[str] | None = None,
    bib_file: str = "references",
) -> str:
    """Convert markdown into a complete compilable LaTeX document."""
    # 先准备元信息，避免后面渲染过程再回头扫描原文。
    final_title = title or extract_title(markdown_text)
    final_authors = authors or []
    abstract = extract_abstract(markdown_text)
    body = build_body(markdown_text)
    # 然后按 LaTeX 文档真实顺序拼装 preamble / abstract / body / references。
    parts = [render_preamble(template, final_title, final_authors)]
    if abstract:
        parts.append("\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n")
    parts.append(body)
    parts.append(render_footer(template, bib_file))
    return "\n".join(part for part in parts if part).strip() + "\n"

# 主转换函数：将 Markdown 文本转成完整的 LaTeX 文档字符串，可直接写入 paper.tex。


def copy_style_files(template: ConferenceTemplate, out_dir: Path) -> None:
    """Copy local .sty/.bst helpers into the LaTeX working directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for style_file in template.get_style_files():
        if style_file.exists():
            shutil.copy2(style_file, out_dir / style_file.name)

# 将模板相关的本地样式文件复制到输出目录，保证编译时能找到 .sty/.bst 等文件。


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one command and capture output for later diagnostics."""
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )

# 运行一个外部命令并捕获输出，用于 pdflatex / bibtex 等调用的诊断。


def parse_latex_errors(log_text: str) -> list[str]:
    """Extract only the most useful LaTeX error lines for debugging."""
    errors: list[str] = []
    for line in log_text.splitlines():
        if line.startswith("! "):
            errors.append(line.strip())
    return errors

# 从 LaTeX 日志中抽取以 "! " 开头的错误信息，作为快速诊断的摘要。


def compile_latex(tex_path: Path, max_attempts: int = 2) -> CompileResult:
    """Compile paper.tex with pdflatex and bibtex if they are available."""
    work_dir = tex_path.parent
    stem = tex_path.stem
    commands: list[list[str]] = []
    errors: list[str] = []
    log_path = work_dir / f"{stem}.log"
    pdf_path = work_dir / f"{stem}.pdf"
    for attempt in range(max_attempts):
        # 第一轮 pdflatex 先生成 .aux / .log，供后续 bibtex 和交叉引用使用。
        pdflatex = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
        commands.append(pdflatex)
        first_run = run_command(pdflatex, work_dir)
        combined_log = (first_run.stdout or "") + "\n" + (first_run.stderr or "")
        if log_path.exists():
            combined_log += "\n" + log_path.read_text(encoding="utf-8", errors="replace")
        errors.extend(parse_latex_errors(combined_log))
        if first_run.returncode != 0:
            continue
        aux_path = work_dir / f"{stem}.aux"
        if aux_path.exists():
            # 有 .aux 才有必要跑 bibtex；否则说明正文里可能没有 bibliography。
            bibtex = ["bibtex", stem]
            commands.append(bibtex)
            bibtex_run = run_command(bibtex, work_dir)
            combined_bibtex_log = (bibtex_run.stdout or "") + "\n" + (bibtex_run.stderr or "")
            errors.extend(parse_latex_errors(combined_bibtex_log))
        second_run = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
        third_run = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
        # 连跑两次是 LaTeX 常规做法，用来稳定目录、引用和页码。
        commands.extend([second_run, third_run])
        second_result = run_command(second_run, work_dir)
        third_result = run_command(third_run, work_dir)
        if second_result.returncode == 0 and third_result.returncode == 0 and pdf_path.exists():
            return CompileResult(
                success=True,
                pdf_path=pdf_path,
                log_path=log_path if log_path.exists() else None,
                commands=commands,
                errors=errors,
            )
    return CompileResult(
        success=False,
        pdf_path=pdf_path if pdf_path.exists() else None,
        log_path=log_path if log_path.exists() else None,
        commands=commands,
        errors=errors,
    )

# 调用 pdflatex/bibtex 的编译入口：尝试若干次并收集命令与错误，返回封装结果。


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a standalone CLI for markdown-to-LaTeX export."""
    parser = argparse.ArgumentParser(description="Render markdown paper to LaTeX/PDF")
    parser.add_argument("--markdown", required=True, help="Input markdown file")
    parser.add_argument("--out-dir", required=True, help="Output directory for paper.tex")
    parser.add_argument("--template", default="generic", help="Template name")
    parser.add_argument("--bib-file", default="references", help="Bib file name without extension")
    parser.add_argument("--author", action="append", default=[], help="Author name; repeatable")
    parser.add_argument("--compile", action="store_true", help="Compile with pdflatex after writing paper.tex")
    return parser

# 小型 CLI 的参数解析器，便于在没有外部框架时独立运行模块。


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint that writes paper.tex and optionally compiles it."""
    args = build_arg_parser().parse_args(argv)
    markdown_text = Path(args.markdown).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    template = get_template(args.template)
    copy_style_files(template, out_dir)
    tex_text = markdown_to_latex(
        markdown_text,
        template,
        authors=args.author,
        bib_file=args.bib_file,
    )
    tex_path = out_dir / "paper.tex"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex_text, encoding="utf-8")
    if args.compile:
        result = compile_latex(tex_path)
        if not result.success:
            for item in result.errors:
                print(item)
            return 1
    return 0

# 主程序：写入 paper.tex 并可选调用 LaTeX 工具链编译，遇到错误会打印简化的错误信息并返回非零码。


if __name__ == "__main__":
    raise SystemExit(main())
