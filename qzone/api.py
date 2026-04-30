"""
QQ 空间 HTTP API 封装 —— 说说列表、点赞、评论、发布
"""
import logging
import time
from typing import Any

from .client import QzoneHttpClient
from .model import ApiResponse
from .session import QzoneSession

logger = logging.getLogger(__name__)


class QzoneAPI(QzoneHttpClient):
    """QQ 空间 HTTP API"""

    BASE_URL = "https://user.qzone.qq.com"
    LIST_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msglist_v6"
    DOLIKE_URL = "https://user.qzone.qq.com/proxy/domain/w.qzone.qq.com/cgi-bin/likes/internal_dolike_app"
    COMMENT_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qzone.qq.com/cgi-bin/emotion_cgi_re_feeds"
    EMOTION_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qzone.qq.com/cgi-bin/emotion_cgi_publish_v6"
    VISITOR_URL = "https://h5.qzone.qq.com/proxy/domain/g.qzone.qq.com/cgi-bin/friendshow/cgi_get_visitor_more"
    DETAIL_URL = "https://h5.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msgdetail_v6"
    DELETE_URL = "https://h5.qzone.qq.com/proxy/domain/taotao.qzone.qq.com/cgi-bin/emotion_cgi_delete_v6"

    def __init__(self, session: QzoneSession, timeout: float = 10.0):
        super().__init__(session, timeout)

    # ==================== 说说列表 ====================

    async def get_feeds(
        self,
        target_id: str,
        *,
        pos: int = 0,
        num: int = 1,
    ) -> ApiResponse:
        """获取指定用户的说说列表"""
        ctx = await self.session.get_ctx()
        raw = await self.request(
            "GET",
            self.LIST_URL,
            params={
                "g_tk": ctx.gtk2,
                "uin": target_id,
                "ftype": 0,
                "sort": 0,
                "pos": pos,
                "num": num,
                "replynum": 100,
                "format": "json",
                "need_comment": 1,
            },
        )
        resp = ApiResponse.from_raw(raw)
        # QZone API 返回格式: {"code":0, "data":{"msglist":[...]}}
        # 把内层 data 展开到 resp.data 方便调用方直接 .get("msglist")
        if resp.ok and isinstance(raw.get("data"), dict):
            resp.data = raw["data"]
        else:
            if isinstance(raw.get("data"), dict):
                _inner = raw["data"]
                logger.warning(f"get_feeds: inner data has_msglist={'msglist' in _inner}, msglist_is_null={_inner.get('msglist') is None}")
        return resp

    async def get_detail(self, tid: str, uin: int) -> ApiResponse:
        """获取单条说说详情"""
        ctx = await self.session.get_ctx()
        raw = await self.request(
            "GET",
            self.DETAIL_URL,
            params={
                "uin": uin,
                "tid": tid,
                "format": "jsonp",
                "g_tk": ctx.gtk2,
            },
        )
        return ApiResponse.from_raw(raw)

    # ==================== 点赞 ====================

    async def like(self, uin: int, tid: str) -> ApiResponse:
        """点赞指定说说"""
        ctx = await self.session.get_ctx()
        raw = await self.request(
            "POST",
            self.DOLIKE_URL,
            params={"g_tk": ctx.gtk2},
            data={
                "qzreferrer": f"{self.BASE_URL}/{ctx.uin}",
                "opuin": ctx.uin,
                "unikey": f"{self.BASE_URL}/{uin}/mood/{tid}",
                "curkey": f"{self.BASE_URL}/{uin}/mood/{tid}",
                "appid": 311,
                "from": 1,
                "typeid": 0,
                "abstime": int(time.time()),
                "fid": tid,
                "active": 0,
                "format": "json",
                "fupdate": 1,
            },
        )
        return ApiResponse.from_raw(raw)

    # ==================== 评论 ====================

    async def comment(self, uin: int, tid: str, content: str) -> ApiResponse:
        """评论指定说说"""
        ctx = await self.session.get_ctx()
        raw = await self.request(
            "POST",
            self.COMMENT_URL,
            params={"g_tk": ctx.gtk2},
            data={
                "topicId": f"{uin}_{tid}__1",
                "uin": ctx.uin,
                "hostUin": uin,
                "feedsType": 100,
                "inCharset": "utf-8",
                "outCharset": "utf-8",
                "plat": "qzone",
                "source": "ic",
                "platformid": 52,
                "format": "fs",
                "ref": "feeds",
                "content": content,
            },
        )
        return ApiResponse.from_raw(raw)

    # ==================== 发布说说 ====================

    async def publish(self, text: str) -> ApiResponse:
        """发表说说"""
        ctx = await self.session.get_ctx()
        raw = await self.request(
            "POST",
            self.EMOTION_URL,
            params={"g_tk": ctx.gtk2, "uin": ctx.uin},
            data={
                "syn_tweet_verson": "1",
                "paramstr": "1",
                "who": "1",
                "con": text,
                "feedversion": "1",
                "ver": "1",
                "ugc_right": "1",
                "to_sign": "0",
                "hostuin": ctx.uin,
                "code_version": "1",
                "format": "json",
                "qzreferrer": f"{self.BASE_URL}/{ctx.uin}",
            },
        )
        return ApiResponse.from_raw(raw)

    # ==================== 删除说说 ====================

    async def delete_post(self, tid: str, feeds_type: int = 1) -> ApiResponse:
        """删除指定说说"""
        ctx = await self.session.get_ctx()
        raw = await self.request(
            "POST",
            self.DELETE_URL,
            params={"g_tk": ctx.gtk2},
            data={
                "uin": ctx.uin,
                "topicId": f"{ctx.uin}_{tid}__1",
                "feedsType": feeds_type,
                "feedsFlag": 0,
                "feedsKey": tid,
                "feedsAppid": 311,
                "feedsTime": int(time.time()),
                "fupdate": 1,
                "ref": "feeds",
                "qzreferrer": (
                    "https://user.qzone.qq.com/"
                    f"proxy/domain/ic2.qzone.qq.com/cgi-bin/feeds/"
                    f"feeds_html_module?g_iframeUser=1&i_uin={ctx.uin}&i_login_uin={ctx.uin}"
                    "&mode=4&previewV8=1&style=35&version=8"
                    "&needDelOpr=true"
                ),
            },
        )
        return ApiResponse.from_raw(raw)

    # ==================== 访客 ====================

    async def get_visitor(self) -> ApiResponse:
        """获取访客"""
        ctx = await self.session.get_ctx()
        raw = await self.request(
            "GET",
            self.VISITOR_URL,
            params={
                "uin": ctx.uin,
                "mask": 7,
                "g_tk": ctx.gtk2,
                "page": 1,
                "fupdate": 1,
                "clear": 1,
            },
        )
        return ApiResponse.from_raw(raw)
