"""
音乐点歌工具 —— 基于 txqq 聚合搜索 API，支持 13+ 音源。

工具只做搜索 + 返回结果标记，由 reply_engine.py 解析标记后发音乐卡片。
避免跨会话并发时 session_info 被覆盖的问题。
"""
import requests
import json
import os
import re
from langchain.tools import tool

# txqq 聚合搜索 API（与 astrbot_plugin_music 同源）
TXQQ_API = "https://music.txqq.pro/"

# 搜索平台映射
PLATFORM_MAP = {
    "netease": "网易云音乐",
    "qq": "QQ音乐",
    "kugou": "酷狗音乐",
    "kuwo": "酷我音乐",
    "baidu": "百度音乐",
    "1ting": "一听音乐",
    "migu": "咪咕音乐",
    "lizhi": "荔枝FM",
    "qingting": "蜻蜓FM",
    "ximalaya": "喜马拉雅",
    "5singyc": "5sing原创",
    "5singfc": "5sing翻唱",
    "kg": "全民K歌",
}

PLATFORM_KEYWORDS = {
    "qq": ["qq", "QQ"],
    "kugou": ["酷狗"],
    "kuwo": ["酷我"],
    "baidu": ["百度"],
    "migu": ["咪咕"],
    "lizhi": ["荔枝"],
    "qingting": ["蜻蜓"],
    "ximalaya": ["喜马"],
    "5singyc": ["5sing原创", "5sing"],
    "5singfc": ["5sing翻唱"],
    "kg": ["全民"],
}

# Music card 类型映射（napcat onebot11 支持的平台）
CARD_TYPE_MAP = {
    "netease": "163",
    "qq": "qq",
    "kugou": "kugou",
    "kuwo": "kuwo",
    "migu": "migu",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://music.txqq.pro",
    "Referer": "https://music.txqq.pro",
}


def _detect_platform(song_name: str) -> str:
    """从歌名中检测用户指定的平台，默认用网易云"""
    for ptype, keywords in PLATFORM_KEYWORDS.items():
        for kw in keywords:
            if kw in song_name:
                return ptype
    return "netease"


def _search_songs(song_name: str, platform: str = "netease", limit: int = 5) -> list[dict]:
    """搜索歌曲，返回歌曲列表"""
    data = {
        "input": song_name,
        "filter": "name",
        "type": platform,
        "page": 1,
    }
    try:
        resp = requests.post(TXQQ_API, data=data, headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("code") == 200 and "data" in result:
                return result["data"][:limit]
    except Exception as e:
        print(f"[音乐] 搜索失败: {e}")
    return []


# 匹配回复中的音乐卡片标记
MUSIC_CARD_RE = re.compile(r'\[MUSIC_CARD:(\w+):(\d+)\]')


def extract_music_card(text: str) -> tuple[str | None, str | None]:
    """从文本中提取音乐卡片标记，返回 (platform, song_id) 或 (None, None)"""
    m = MUSIC_CARD_RE.search(text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def strip_music_card_marker(text: str) -> str:
    """移除文本中的音乐卡片标记"""
    return MUSIC_CARD_RE.sub("", text).strip()


@tool
def play_music(song_name: str) -> str:
    """
    点歌/搜歌：当用户想听歌、搜某首歌时使用。
    根据歌曲名称（可含歌手名）搜索音乐，支持网易云、QQ音乐、酷狗等多平台。
    如果找到了歌曲，会附加音乐卡片标记供系统发送。

    用法示例：
    - "点歌 十年" → 搜索并播放"十年"
    - "来一首周杰伦的七里香" → 搜索并播放"七里香"
    - "酷狗点歌 泡沫" → 会自动识别"酷狗"平台
    """
    # 检测平台
    platform = _detect_platform(song_name)
    # 清理输入
    cleanup_words = ["点歌", "听歌", "放歌", "搜歌", "播放", "来一首", "放一首"]
    for keywords in PLATFORM_KEYWORDS.values():
        for kw in keywords:
            if kw in song_name:
                song_name = song_name.replace(kw, "").strip()
    for cw in cleanup_words:
        if song_name.startswith(cw):
            song_name = song_name[len(cw):].strip()
        if song_name.endswith(cw):
            song_name = song_name[:-len(cw)].strip()
    clean_name = song_name
    if not clean_name:
        clean_name = "未知歌曲"

    platform_display = PLATFORM_MAP.get(platform, platform)

    songs = _search_songs(clean_name, platform)
    if not songs:
        return f"在{platform_display}上没有找到「{clean_name}」的相关歌曲"

    first = songs[0]
    song_id = first.get("songid")
    song_title = first.get("title", "未知")
    song_artist = first.get("author", "未知")
    card_type = CARD_TYPE_MAP.get(platform)

    # 如果支持音乐卡片，附加标记
    if song_id and card_type:
        marker = f"[MUSIC_CARD:{platform}:{song_id}]"
        return f"已为你播放 {song_title} - {song_artist} {marker}"

    # 不支持卡片的平台，返回链接让 LLM 自己处理
    song_url = first.get("url") or first.get("link", "")
    result = f"已找到 {song_title} - {song_artist}"
    if song_url:
        result += f"\n播放链接: {song_url}"
    return result
