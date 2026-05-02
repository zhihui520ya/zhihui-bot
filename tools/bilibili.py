# bilibili_tools.py
"""B站视频搜索/热门/详情工具，使用 requests 直接调用 B 站公开 API"""

import requests
from langchain.tools import tool

BILI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com",
}


def _fmt_count(n: int) -> str:
    """将数字格式化为万/亿单位"""
    if n >= 100000000:
        return f"{n / 100000000:.1f}亿"
    if n >= 10000:
        return f"{n / 10000:.1f}万"
    return str(n)


def _sec_to_duration(seconds: int) -> str:
    """将秒数转为 mm:ss 或 hh:mm:ss"""
    m, s = divmod(seconds, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _bili_api_get(url: str, params: dict | None = None) -> dict | None:
    """通用 B 站 API GET 请求，返回 data 字段或 None"""
    try:
        resp = requests.get(url, headers=BILI_HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        print(f"[B站工具] 请求失败 {url}: {e}")
        return None

    if body.get("code") != 0:
        print(f"[B站工具] API 返回错误 code={body.get('code')}, msg={body.get('message')}")
        return None
    return body.get("data")


@tool
def search_bilibili_video(keyword: str, page: int = 1) -> str:
    """搜索B站视频。输入关键词，返回视频列表。"""
    print(f"[工具调用] search_bilibili_video(keyword={keyword}, page={page})")

    data = _bili_api_get(
        "https://api.bilibili.com/x/web-interface/search/type",
        {"search_type": "video", "keyword": keyword, "page": page, "page_size": 10},
    )
    if not data:
        return f"搜索“{keyword}”时出错了，可能是网络问题，稍后再试试吧~"

    results = data.get("result", [])
    if not results:
        return f"没搜到“{keyword}”相关视频，换个关键词试试？"

    lines = [f"🔍 搜索“{keyword}”的结果（共{data.get('numResults', 0)}条）:\n"]
    for i, v in enumerate(results[:10], 1):
        title = v.get("title", "")
        # B 站 API 返回的 title 带 <em> 高亮标签，去掉
        title = title.replace("<em class=\"keyword\">", "").replace("</em>", "")
        author = v.get("author", "未知")
        play = _fmt_count(v.get("play", 0))
        like = _fmt_count(v.get("like", 0))
        duration = v.get("duration", "?")
        bvid = v.get("bvid", "")
        desc = (v.get("description", "") or "")[:60]
        lines.append(
            f"{i}. {title}\n"
            f"   UP主：{author} | 播放：{play} | 点赞：{like} | 时长：{duration}\n"
            f"   BV号：{bvid}\n"
            f"   简介：{desc}\n"
        )

    return "\n".join(lines)


@tool
def get_bilibili_hot() -> str:
    """获取B站当前热门视频。"""
    print("[工具调用] get_bilibili_hot()")

    data = _bili_api_get(
        "https://api.bilibili.com/x/web-interface/popular",
        {"pn": 1, "ps": 10},
    )
    if not data:
        return "热门视频获取失败了，稍后再试试吧~"

    vlist = data.get("list", [])
    if not vlist:
        return "热门视频是空的，可能是接口出问题了~"

    lines = ["🔥 B站当前热门视频:\n"]
    for i, v in enumerate(vlist[:10], 1):
        title = v.get("title", "无标题")
        owner = v.get("owner", {})
        author = owner.get("name", "未知")
        stat = v.get("stat", {})
        play = _fmt_count(stat.get("view", 0))
        like = _fmt_count(stat.get("like", 0))
        danmaku = _fmt_count(stat.get("danmaku", 0))
        duration = _sec_to_duration(v.get("duration", 0))
        bvid = v.get("bvid", "")
        lines.append(
            f"{i}. {title}\n"
            f"   UP主：{author} | 播放：{play} | 点赞：{like} | 弹幕：{danmaku} | 时长：{duration}\n"
            f"   BV号：{bvid}\n"
        )

    return "\n".join(lines)


@tool
def get_video_info(bvid: str) -> str:
    """获取B站视频的详细信息。输入BV号（如 BV1GJ411x7Gt）。"""
    print(f"[工具调用] get_video_info(bvid={bvid})")

    data = _bili_api_get(
        "https://api.bilibili.com/x/web-interface/view",
        {"bvid": bvid},
    )
    if not data:
        return f"获取视频 {bvid} 的信息失败了，检查一下BV号是否正确？"

    title = data.get("title", "无标题")
    desc = data.get("desc", "无简介") or "无简介"
    owner = data.get("owner", {})
    author = owner.get("name", "未知")
    author_uid = owner.get("mid", "?")
    stat = data.get("stat", {})
    play = _fmt_count(stat.get("view", 0))
    like = _fmt_count(stat.get("like", 0))
    coin = _fmt_count(stat.get("coin", 0))
    favorite = _fmt_count(stat.get("favorite", 0))
    share = _fmt_count(stat.get("share", 0))
    danmaku = _fmt_count(stat.get("danmaku", 0))
    reply = _fmt_count(stat.get("reply", 0))
    duration = _sec_to_duration(data.get("duration", 0))
    tid = data.get("tid", 0)
    tname = data.get("tname", "未知分区")

    return (
        f"📺 {title}\n"
        f"UP主：{author}（UID: {author_uid}）\n"
        f"分区：{tname}\n"
        f"时长：{duration}\n\n"
        f"📊 数据概况\n"
        f"播放：{play} | 点赞：{like} | 硬币：{coin}\n"
        f"收藏：{favorite} | 转发：{share} | 弹幕：{danmaku} | 评论：{reply}\n\n"
        f"📝 简介\n{desc}\n"
    )
