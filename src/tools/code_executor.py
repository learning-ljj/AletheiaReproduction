"""Python 代码沙箱执行器：通过 subprocess 隔离执行代码。"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# 沙箱子进程环境：强制 UTF-8 输出，避免 Windows GBK 编码错误
_SANDBOX_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run_python(code: str, timeout: int = 30) -> dict:
    """执行 Python 代码字符串。

    返回 {"stdout": "...", "stderr": "...", "exit_code": 0}
    超时则 exit_code 非零，stderr 包含超时提示。
    """
    # 将代码写入临时文件后执行，避免命令行转义问题
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write(code)
        tmp.close()

        result = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True,
            text=True,
            encoding="utf-8",   # 显式 UTF-8 解码，避免 Windows GBK 误解析
            timeout=timeout,
            env=_SANDBOX_ENV,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        # Windows 上 TimeoutExpired 处理后可能残留子进程，显式 kill
        if hasattr(exc, "process") and exc.process is not None:
            try:
                exc.process.kill()
            except OSError:
                pass
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "exit_code": 1,
        }
    except (KeyboardInterrupt, OSError) as exc:
        # Windows 下 subprocess timeout 偶发 KeyboardInterrupt / OSError
        return {
            "stdout": "",
            "stderr": f"Execution interrupted: {type(exc).__name__}: {exc}",
            "exit_code": 1,
        }
    finally:
        Path(tmp.name).unlink(missing_ok=True)
