"""
QZone HTTP 客户端 —— httpx 封装，自动重登
"""
import logging
from typing import Any

import httpx

from .constants import (
    HTTP_STATUS_FORBIDDEN,
    HTTP_STATUS_UNAUTHORIZED,
    QZONE_CODE_LOGIN_EXPIRED,
    QZONE_CODE_UNKNOWN,
    QZONE_INTERNAL_HTTP_STATUS_KEY,
    QZONE_INTERNAL_META_KEY,
    QZONE_MSG_PERMISSION_DENIED,
)
from .model import QzoneContext
from .parser import QzoneParser
from .session import QzoneSession

logger = logging.getLogger(__name__)


class QzoneHttpClient:
    def __init__(self, session: QzoneSession, timeout: float = 10.0):
        self.session = session
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def close(self):
        await self._http.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
        retry: int = 0,
    ) -> dict[str, Any]:
        ctx = await self.session.get_ctx()

        kwargs = {
            "params": params,
            "data": data,
            "headers": headers or ctx.headers(),
        }
        if timeout:
            kwargs["timeout"] = timeout

        resp = await self._http.request(method, url, **kwargs)
        text = resp.text

        # 调试：记录 QZone API 请求和响应
        if "emotion_cgi_msglist" in url or "emotion_cgi_delete" in url or "emotion_cgi_publish" in url:
            sent_headers = dict(kwargs.get("headers", {}))
            cookie_preview = sent_headers.get("Cookie", "(none)")[:60]
            print(f"[QZone请求调试] Cookie前60字: {cookie_preview}...")
            print(f"[QZone请求调试] {method} {url}")
            if data:
                data_preview = str(data)[:200]
                print(f"[QZone请求调试] POST数据: {data_preview}")
            print(f"[QZone请求调试] 状态码={resp.status_code}")
            print(f"[QZone请求调试] 原始响应前200字: {text[:200]}")

        parsed = QzoneParser.parse_response(text)

        if "emotion_cgi_msglist" in url:
            _msglist = parsed.get("msglist")
            _count = len(_msglist) if isinstance(_msglist, list) else 0
            print(f"[QZone请求调试] 解析后 code={parsed.get('code')}, msglist条数={_count}")
            print(f"[QZone请求调试] 解析后 keys={list(parsed.keys())}")
        elif "emotion_cgi_delete" in url:
            print(f"[QZone请求调试] 删除响应 code={parsed.get('code')}, message={parsed.get('message', '')}")
        elif "emotion_cgi_publish" in url:
            print(f"[QZone请求调试] 发布响应 code={parsed.get('code')}, message={parsed.get('message', '')}")

        meta = parsed.get(QZONE_INTERNAL_META_KEY)
        if not isinstance(meta, dict):
            meta = {}
            parsed[QZONE_INTERNAL_META_KEY] = meta
        meta[QZONE_INTERNAL_HTTP_STATUS_KEY] = resp.status_code

        # 登录失效时自动重试
        if resp.status_code == HTTP_STATUS_UNAUTHORIZED or parsed.get(
            "code"
        ) == QZONE_CODE_LOGIN_EXPIRED:
            if retry >= 2:
                raise RuntimeError("登录失效，重试失败")
            logger.warning("QZone 登录失效，请更新 Cookie")
            # 无法自动重新登录（无 aiocqhttp），直接报错
            raise RuntimeError("QZone Cookie 已失效，请更新")

        if resp.status_code == HTTP_STATUS_FORBIDDEN and parsed.get("code") in (
            QZONE_CODE_UNKNOWN,
            None,
        ):
            parsed["code"] = resp.status_code
            parsed["message"] = QZONE_MSG_PERMISSION_DENIED

        return parsed
