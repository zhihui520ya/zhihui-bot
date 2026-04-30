from dataclasses import dataclass
from typing import Any


class QzoneContext:
    """统一封装 Qzone 请求所需的所有动态参数"""

    def __init__(self, uin: int, skey: str, p_skey: str, cookie_str: str = ""):
        self.uin = uin
        self.skey = skey
        self.p_skey = p_skey
        self._cookie_str = cookie_str  # 原始完整 cookie 字符串

    @property
    def gtk2(self) -> str:
        """动态计算 gtk2"""
        hash_val = 5381
        for ch in self.p_skey:
            hash_val += (hash_val << 5) + ord(ch)
        return str(hash_val & 0x7FFFFFFF)

    def cookies(self) -> dict[str, str]:
        return {
            "uin": f"o{self.uin}",
            "skey": self.skey,
            "p_skey": self.p_skey,
        }

    @property
    def cookie_header(self) -> str:
        """返回完整 Cookie 字符串（原始格式，用于请求头）"""
        return self._cookie_str

    def headers(self) -> dict[str, str]:
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "referer": f"https://user.qzone.qq.com/{self.uin}",
            "origin": "https://user.qzone.qq.com",
        }
        if self._cookie_str:
            hdrs["Cookie"] = self._cookie_str
        return hdrs


@dataclass
class ApiResponse:
    """统一接口响应结果"""

    ok: bool
    code: int
    message: str | None
    data: dict[str, Any]
    raw: dict[str, Any]

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        *,
        code_key: str = "code",
        msg_key: str | tuple[str, ...] = ("message", "msg"),
        data_key: str | None = None,
        success_code: int = 0,
    ) -> "ApiResponse":
        code = raw.get(code_key, -1)
        message = None
        if isinstance(msg_key, tuple):
            for k in msg_key:
                if raw.get(k):
                    message = raw.get(k)
                    break
        else:
            message = raw.get(msg_key)
        if code == success_code:
            data = dict(raw)
            data.pop("__qzone_internal__", None)
            return cls(ok=True, code=code, message=None, data=data, raw=raw)
        return cls(ok=False, code=code, message=message, data={}, raw=raw)

    def __bool__(self) -> bool:
        return self.ok

    def get(self, key: str, default: Any = None) -> Any:
        if not self.ok or not self.data:
            return default
        return self.data.get(key, default)
