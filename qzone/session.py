"""
QQ 空间登录上下文 —— Cookie 解析 + gtk2 计算
"""
import asyncio
import logging
from http.cookies import SimpleCookie

from .model import QzoneContext

logger = logging.getLogger(__name__)


class QzoneSession:
    """QQ 登录上下文，从 Cookie 字符串解析登录态"""

    DOMAIN = "user.qzone.qq.com"

    def __init__(self, cookies_str: str):
        self._cookies_str = cookies_str
        self._ctx: QzoneContext | None = None
        self._lock = asyncio.Lock()

    async def get_ctx(self) -> QzoneContext:
        async with self._lock:
            if not self._ctx:
                self._ctx = self._parse_cookies(self._cookies_str)
            return self._ctx

    async def get_uin(self) -> int:
        ctx = await self.get_ctx()
        return ctx.uin

    def _parse_cookies(self, cookies_str: str) -> QzoneContext:
        c = {k: v.value for k, v in SimpleCookie(cookies_str).items()}
        uin_raw = c.get("uin", "o0")
        uin = int(uin_raw[1:]) if uin_raw.startswith("o") else int(uin_raw)
        if not uin:
            raise RuntimeError("Cookie 中缺少合法 uin")
        ctx = QzoneContext(
            uin=uin,
            skey=c.get("skey", ""),
            p_skey=c.get("p_skey", ""),
            cookie_str=cookies_str,
        )
        logger.info(f"QZone 登录成功，uin={uin}")
        return ctx
