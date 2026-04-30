"""
QQ 空间响应解析器 —— JSONP 解析、说说列表解析、访客解析
"""
import json
import logging
import re
from typing import Any

import json5

from .constants import (
    QZONE_CODE_UNKNOWN,
    QZONE_MSG_EMPTY_RESPONSE,
    QZONE_MSG_INVALID_RESPONSE,
    QZONE_MSG_JSON_PARSE_ERROR,
    QZONE_MSG_NON_OBJECT_RESPONSE,
)

logger = logging.getLogger(__name__)


class QzoneParser:
    """QQ 空间响应解析器"""

    @staticmethod
    def _error_payload(message: str) -> dict[str, Any]:
        return {"code": QZONE_CODE_UNKNOWN, "message": message, "data": {}}

    @staticmethod
    def parse_response(text: str) -> dict[str, Any]:
        """解析 JSON / JSONP / 非标准 JSON"""
        if not text or not text.strip():
            logger.warning("响应内容为空")
            return QzoneParser._error_payload(QZONE_MSG_EMPTY_RESPONSE)

        if m := re.search(
            r"callback\s*\(\s*([^{]*(\{.*\})[^)]*)\s*\)",
            text,
            re.I | re.S,
        ):
            json_str = m.group(2)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end < start:
                logger.warning("响应内容缺少 JSON 片段")
                return QzoneParser._error_payload(QZONE_MSG_INVALID_RESPONSE)
            json_str = text[start: end + 1]

        json_str = json_str.replace("undefined", "null").strip()

        try:
            data = json5.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"JSON 解析错误: {e}")
            return QzoneParser._error_payload(QZONE_MSG_JSON_PARSE_ERROR)

        if not isinstance(data, dict):
            logger.error("JSON 解析结果不是 dict")
            return QzoneParser._error_payload(QZONE_MSG_NON_OBJECT_RESPONSE)

        return data

    @staticmethod
    def parse_feeds(msglist: list[dict]) -> list[dict]:
        """解析说说列表为简单 dict 列表"""
        posts = []
        for msg in msglist:
            image_urls = []
            for img_data in msg.get("pic", []):
                for key in ("url2", "url3", "url1"):
                    if raw := img_data.get(key):
                        image_urls.append(raw)
                        break
            for video in msg.get("video") or []:
                video_image_url = video.get("url1") or video.get("pic_url")
                if video_image_url:
                    image_urls.append(video_image_url)
            rt_con = msg.get("rt_con", {}).get("content", "")
            post = {
                "tid": msg.get("tid", ""),
                "uin": msg.get("uin", 0),
                "name": msg.get("name", ""),
                "text": msg.get("content", "").strip(),
                "images": image_urls,
                "create_time": msg.get("created_time", 0),
                "feedstype": msg.get("feedstype", 1),
                "rt_con": rt_con,
                "comments": [
                    {
                        "uin": c.get("uin", 0),
                        "nickname": c.get("name", ""),
                        "content": c.get("content", ""),
                    }
                    for c in msg.get("commentlist") or []
                ],
            }
            posts.append(post)
        return posts
