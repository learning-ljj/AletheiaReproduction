"""共享 HTTP 工具：SSL 降级重试封装（供 web_search 和 wiki_search 复用）。"""

import logging
import ssl
import time
import urllib.error
import urllib.request

_logger = logging.getLogger(__name__)
_MAX_FETCH_RETRIES = 3


def _make_lenient_ssl_ctx() -> ssl.SSLContext:
    """创建宽松 SSL 上下文，用于 SSL 握手失败的降级重试。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def urlopen_with_retry(req: urllib.request.Request, timeout: int) -> bytes:
    """带重试与 SSL 降级的 urlopen 封装。

    重试策略：
      1. 首次使用默认 SSL（标准安全）；
      2. 若 SSL 错误（含 URLError 包裹的证书校验失败），降级到宽松 SSL 上下文重试；
      3. 其他网络错误：最多 _MAX_FETCH_RETRIES 次指数退避。
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_FETCH_RETRIES):
        if attempt > 0:
            time.sleep(2 ** (attempt - 1))  # 1 s → 2 s 指数退避
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except ssl.SSLError as e:
            _logger.warning("SSL error on attempt %d/%d: %s — retrying with lenient SSL context.",
                            attempt + 1, _MAX_FETCH_RETRIES, e)
            try:
                with urllib.request.urlopen(req, timeout=timeout,
                                            context=_make_lenient_ssl_ctx()) as resp:
                    return resp.read()
            except Exception as e2:
                last_error = e2
                _logger.warning("Lenient SSL retry also failed: %s", e2)
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", None)
            if isinstance(reason, ssl.SSLError):
                _logger.warning(
                    "SSL URLError on attempt %d/%d: %s — retrying with lenient SSL context.",
                    attempt + 1, _MAX_FETCH_RETRIES, e,
                )
                try:
                    with urllib.request.urlopen(req, timeout=timeout,
                                                context=_make_lenient_ssl_ctx()) as resp:
                        return resp.read()
                except Exception as e2:
                    last_error = e2
                    _logger.warning("Lenient SSL retry also failed: %s", e2)
            else:
                last_error = e
                _logger.warning("Network error on attempt %d/%d: %s", attempt + 1, _MAX_FETCH_RETRIES, e)
        except Exception as e:
            last_error = e
            _logger.warning("Network error on attempt %d/%d: %s", attempt + 1, _MAX_FETCH_RETRIES, e)
    raise last_error or RuntimeError("urlopen_with_retry: unexpected failure")
