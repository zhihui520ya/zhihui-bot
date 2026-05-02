"""
QQ 空间 LangChain 工具 —— LLM 可在对话中自主调用看说说/点赞/评论/发说说
Cookie 自动从 NapCat 获取，无需手动配置
"""
import asyncio
import logging
import time

import httpx
from langchain.tools import tool

from config import QZONE_COOKIES, NAPCAT_BASE_URL, NAPCAT_TOKEN
from qzone import QzoneAPI, QzoneParser, QzoneSession

logger = logging.getLogger(__name__)


def _run_async(coro):
    """在同步上下文（线程池）中执行异步协程"""
    return asyncio.run(coro)


# Session 缓存：复用同一个 QzoneSession 对象，保持 gtk2 一致
_cached_session: QzoneSession | None = None
_cached_session_time: float = 0
SESSION_CACHE_TTL = 300  # 缓存 5 分钟


def _invalidate_session_cache():
    """强制清除 session 缓存，下次调用 _ensure_session 会重新获取 cookie"""
    global _cached_session, _cached_session_time
    _cached_session = None
    _cached_session_time = 0
    print("[QZone] Session 缓存已清除")


def _ensure_session() -> QzoneSession | None:
    """获取 QZone Session（缓存 QzoneSession 对象，保持 gtk2 一致）"""
    global _cached_session, _cached_session_time
    now = time.time()
    if _cached_session and (now - _cached_session_time) < SESSION_CACHE_TTL:
        logger.debug("使用缓存的 QZone Session")
        return _cached_session

    # 从 NapCat 获取 Cookie，失败时回退到 config 中的静态 Cookie
    cookies: str | None = None
    try:
        resp = httpx.get(
            f"{NAPCAT_BASE_URL}/get_cookies",
            params={"domain": "user.qzone.qq.com"},
            headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
            timeout=5,
        )
        data = resp.json()
        cookies = data.get("data", {}).get("cookies", "")
        if cookies:
            logger.info("已从 NapCat 获取 QZone Cookie")
        else:
            logger.warning(f"NapCat 返回的 Cookie 为空: {data}")
    except Exception as e:
        logger.warning(f"从 NapCat 获取 Cookie 失败: {e}")

    if not cookies:
        cookies = QZONE_COOKIES
    if not cookies:
        return None

    session = QzoneSession(cookies)
    _cached_session = session
    _cached_session_time = now
    return session


def _resolve_user_id(user_id: str, user_name: str = "") -> str | None:
    """通过 QQ 号或群昵称/名称解析目标用户 QQ 号"""
    if user_id:
        return user_id
    if not user_name:
        return None

    try:
        resp = httpx.get(
            f"{NAPCAT_BASE_URL}/get_group_list",
            headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
            timeout=5,
        )
        groups = resp.json().get("data", [])
    except Exception as e:
        logger.warning(f"获取群列表失败: {e}")
        return None

    user_name_lower = user_name.strip().lower()

    # 角色关键词检测：当说"群主""管理员"时按角色查找
    role_keyword_map = {
        "群主": "owner",
        "管理员": "admin",
        "管理": "admin",
    }
    target_role = None
    for keyword, role in role_keyword_map.items():
        if keyword in user_name_lower:
            target_role = role
            break

    for group in groups:
        gid = group.get("group_id")
        if not gid:
            continue
        try:
            mr = httpx.get(
                f"{NAPCAT_BASE_URL}/get_group_member_list",
                params={"group_id": gid},
                headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
                timeout=5,
            )
            members = mr.json().get("data", [])
            # 按角色查找
            if target_role:
                for m in members:
                    if m.get("role") == target_role:
                        name = (m.get("card") or m.get("nickname") or "").strip()
                        return str(m.get("user_id", ""))
            # 按昵称查找
            for m in members:
                name = (m.get("card") or m.get("nickname") or "").strip().lower()
                if name == user_name_lower or user_name_lower in name:
                    return str(m.get("user_id", ""))
        except Exception as e:
            logger.warning(f"查询群 {gid} 成员失败: {e}")
            continue
    return None


def _fmt_post(post: dict, idx: int = 0) -> str:
    """格式化单条说说为可读文本"""
    lines = [f"【第{idx + 1}条】"]
    lines.append(f"用户: {post.get('name', '未知')}({post.get('uin', '?')})")
    lines.append(f"tid: {post.get('tid', '')}")
    lines.append(f"类型: {post.get('feedstype', '?')}")
    lines.append(f"内容: {post.get('text', '(无)')}")
    if post.get("rt_con"):
        lines.append(f"转发: {post['rt_con']}")
    if post.get("images"):
        lines.append(f"图片: {len(post['images'])} 张")
    comments = post.get("comments", [])
    if comments:
        lines.append(f"评论 ({len(comments)} 条):")
        for c in comments[:5]:
            lines.append(f"  - {c.get('nickname', '')}: {c.get('content', '')}")
        if len(comments) > 5:
            lines.append(f"  ... 还有 {len(comments) - 5} 条")
    return "\n".join(lines)


# ==================== 工具函数 ====================


@tool
def qzone_view_feeds(user_id: str = "", user_name: str = "", pos: int = 0, num: int = 5) -> str:
    """
    查看某个人的 QQ 空间说说/动态。只需要知道对方的群昵称或QQ昵称即可，不需要 QQ 号。
    不知道 QQ 号时传 user_name 参数，支持群昵称、QQ昵称、"群主"、"管理员"。
    如果 user_name 也找不到，可以先调 qzone_lookup_member 查。

    注意：让用户提供 QQ 号是多余的——你直接传 user_name 就能查到。

    Args:
        user_id: 目标 QQ 号（纯数字，不填也可以）
        user_name: 对方的群昵称、QQ昵称、"群主"或"管理员"
        pos: 起始位置，0=最新
        num: 获取条数，默认10
    """
    uid = _resolve_user_id(user_id, user_name)
    if not uid:
        return "不知道你想看谁的空间，请告诉我 QQ 号或对方的群昵称"

    session = _ensure_session()
    if not session:
        return "QQ 空间功能未配置"

    async def _fetch():
        cur_session = session
        ctx = await cur_session.get_ctx()
        list_url = "https://user.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msglist_v6"
        async with httpx.AsyncClient() as client:
            for attempt in range(2):
                if attempt == 1:
                    print(f"[QZone] 首次空 msglist，刷新 cookie 重试 (view_feeds)...")
                    _invalidate_session_cache()
                    new_session = _ensure_session()
                    if not new_session:
                        return f"该用户暂无可见说说"
                    cur_session = new_session
                    ctx = await cur_session.get_ctx()

                feeds_resp = await client.get(list_url, params={
                    "g_tk": ctx.gtk2, "uin": uid, "ftype": 0, "sort": 0,
                    "pos": pos, "num": num, "replynum": 100,
                    "format": "json", "need_comment": 1,
                }, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "referer": f"https://user.qzone.qq.com/{ctx.uin}",
                    "origin": "https://user.qzone.qq.com",
                    "Cookie": ctx.cookie_header,
                })
                parsed = QzoneParser.parse_response(feeds_resp.text)
                if parsed.get("code") != 0:
                    continue
                msglist = parsed.get("msglist") or []
                if msglist:
                    break
            else:
                return f"该用户暂无可见说说"

        posts = QzoneParser.parse_feeds(msglist)
        parts = [f"共 {len(posts)} 条说说："]
        for i, post in enumerate(posts):
            parts.append(_fmt_post(post, i))
        return "\n---\n".join(parts)

    return _run_async(_fetch())


@tool
def qzone_search_post(keyword: str, user_id: str = "", user_name: str = "", num: int = 20) -> str:
    """
    搜索某个人的说说内容。不需要知道 QQ 号，传群昵称/QQ昵称就够。
    可以用来找特定内容的说说，比如"分手"、"520"等。
    找到后返回说说的 tid，可以配合 qzone_comment_post / qzone_like_post 使用。

    Args:
        keyword: 搜索关键词（在说说内容中匹配）
        user_id: 目标 QQ 号（纯数字，不填也可以）
        user_name: 对方的群昵称、QQ昵称、"群主"或"管理员"
        num: 搜索范围（最近多少条说说），默认20
    """
    uid = _resolve_user_id(user_id, user_name)
    if not uid:
        return "不知道你想搜谁的空间，请告诉我 QQ 号或对方的群昵称"

    session = _ensure_session()
    if not session:
        return "QQ 空间功能未配置"

    async def _search():
        cur_session = session
        ctx = await cur_session.get_ctx()
        list_url = "https://user.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msglist_v6"
        async with httpx.AsyncClient() as client:
            all_posts = []
            seen_tids = set()
            current_pos = 0
            while len(all_posts) < num:
                for attempt in range(2):
                    if attempt == 1:
                        if all_posts:
                            break
                        _invalidate_session_cache()
                        new_session = _ensure_session()
                        if not new_session:
                            break
                        cur_session = new_session
                        ctx = await cur_session.get_ctx()

                    feeds_resp = await client.get(list_url, params={
                        "g_tk": ctx.gtk2, "uin": uid, "ftype": 0, "sort": 0,
                        "pos": current_pos, "num": 7, "replynum": 100,
                        "format": "json", "need_comment": 1,
                    }, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "referer": f"https://user.qzone.qq.com/{ctx.uin}",
                        "origin": "https://user.qzone.qq.com",
                        "Cookie": ctx.cookie_header,
                    })
                    parsed = QzoneParser.parse_response(feeds_resp.text)
                    if parsed.get("code") != 0:
                        continue
                    page_msglist = parsed.get("msglist") or []
                    if not page_msglist:
                        break
                    for p in page_msglist:
                        tid = p.get("tid", "")
                        if tid not in seen_tids:
                            seen_tids.add(tid)
                            all_posts.append(p)
                    current_pos += len(page_msglist)
                    total = parsed.get("total", 0)
                    if len(all_posts) >= total:
                        break
                    break
                if not page_msglist or len(all_posts) >= total:
                    break

        if not all_posts:
            return f"{uid} 的说说都搜完了，没找到包含「{keyword}」的"

        posts = QzoneParser.parse_feeds(all_posts)
        matched = [p for p in posts if keyword.lower() in p["text"].lower()]
        if not matched:
            return f"最近 {len(all_posts)} 条说说中没找到包含「{keyword}」的"

        parts = [f"找到 {len(matched)} 条包含「{keyword}」的说说："]
        for i, post in enumerate(matched):
            parts.append(_fmt_post(post, i))
        return "\n---\n".join(parts)

    return _run_async(_search())


@tool
def qzone_like_post(user_id: str = "", user_name: str = "", pos: int = 0, tid: str = "") -> str:
    """
    给某个人的 QQ 空间说说点赞。知道 tid 时直接传 tid，不知道时传 pos（首条=0）。
    只需要知道对方的群昵称或QQ昵称即可，不需要 QQ 号。
    如果 user_name 也找不到，可以先调 qzone_lookup_member 查。

    Args:
        user_id: 目标 QQ 号（纯数字，不填也可以）
        user_name: 对方的群昵称、QQ昵称，或"群主"/"管理员"
        pos: 要点赞的说说是第几条，0=最新（tid 为空时使用）
        tid: 说说 ID（知道时直接传，跳过翻找）
    """
    if tid and not user_id:
        uid = ""
    else:
        uid = _resolve_user_id(user_id, user_name)
    if not uid and not tid:
        return "不知道给谁点赞，请告诉我 QQ 号或群昵称"

    session = _ensure_session()
    if not session:
        return "QQ 空间功能未配置"

    async def _do():
        cur_session = session
        ctx = await cur_session.get_ctx()
        post = None
        tid_to_use = tid
        post_uin = int(uid) if uid else 0

        if tid_to_use:
            # 直接使用传入的 tid，找一下 uin
            detail_api = QzoneAPI(cur_session, timeout=5)
            try:
                detail = await detail_api.get_detail(tid_to_use, post_uin)
                if detail.ok:
                    post = {"tid": tid_to_use, "uin": post_uin, "content": detail.get("content", "")}
            except Exception:
                pass
            finally:
                await detail_api.close()
            if not post:
                post = {"tid": tid_to_use, "uin": post_uin, "content": ""}
        else:
            # 按 pos 找说说
            list_url = "https://user.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msglist_v6"
            async with httpx.AsyncClient() as client:
                for attempt in range(2):
                    if attempt == 1:
                        print(f"[QZone] 首次空 msglist，刷新 cookie 重试 (like_post)...")
                        _invalidate_session_cache()
                        new_session = _ensure_session()
                        if not new_session:
                            return "该用户暂无可见说说"
                        cur_session = new_session
                        ctx = await cur_session.get_ctx()

                    feeds_resp = await client.get(list_url, params={
                        "g_tk": ctx.gtk2, "uin": uid, "ftype": 0, "sort": 0,
                        "pos": pos, "num": 7, "replynum": 100,
                        "format": "json", "need_comment": 1,
                    }, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "referer": f"https://user.qzone.qq.com/{ctx.uin}",
                        "origin": "https://user.qzone.qq.com",
                        "Cookie": ctx.cookie_header,
                    })
                    parsed = QzoneParser.parse_response(feeds_resp.text)
                    if parsed.get("code") != 0:
                        continue
                    msglist = parsed.get("msglist") or []
                    if msglist:
                        break
                else:
                    return "该用户暂无可见说说"

            post = msglist[0]
            tid_to_use = post.get("tid", "")
            post_uin = post.get("uin", 0)
            if not tid_to_use:
                return "说说 ID 为空"

        # 点赞
        async def _like():
            api = QzoneAPI(cur_session)
            try:
                like_resp = await api.like(post_uin, tid_to_use)
                if like_resp.ok:
                    return f"已点赞 {post.get('name', '未知用户')} 的说说：{post.get('content', '')[:50]}"
                return f"点赞失败: {like_resp.message}"
            finally:
                await api.close()
        return await _like()

    return _run_async(_do())


@tool
def qzone_comment_post(content: str, user_id: str = "", user_name: str = "", pos: int = 0, tid: str = "") -> str:
    """
    评论某个人的 QQ 空间说说。当用户要求"评论"时你必须实际调用此工具执行，不能只是口头上答应。
    只需要知道对方的群昵称或QQ昵称即可，不需要 QQ 号。
    不知道 QQ 号时传 user_name 参数，支持群昵称、QQ昵称、"群主"、"管理员"。
    如果 user_name 也找不到，可以先调 qzone_lookup_member 查。

    注意：让用户提供 QQ 号是多余的——你直接传 user_name 就能查到。
    注意：你不能说"好了，评了"然后不调用此工具。必须实际执行调用。

    Args:
        content: 评论内容（必填）
        user_id: 目标 QQ 号（纯数字，不填也可以）
        user_name: 对方的群昵称、QQ昵称，或"群主"/"管理员"
        pos: 第几条说说，0=最新（tid 为空时使用）
        tid: 说说 ID（知道时直接传，跳过翻找）
    """
    if tid and not user_id:
        uid = ""
    else:
        uid = _resolve_user_id(user_id, user_name)
    if not uid and not tid:
        return "不知道评论谁的空间，请告诉我 QQ 号或群昵称"

    session = _ensure_session()
    if not session:
        return "QQ 空间功能未配置"

    async def _do():
        cur_session = session
        ctx = await cur_session.get_ctx()
        post = None
        tid_to_use = tid
        post_uin = int(uid) if uid else 0

        if tid_to_use:
            # 直接使用传入的 tid
            post = {"tid": tid_to_use, "uin": post_uin, "name": "", "content": ""}
        else:
            # 按 pos 找说说
            list_url = "https://user.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msglist_v6"
            async with httpx.AsyncClient() as client:
                for attempt in range(2):
                    if attempt == 1:
                        print(f"[QZone] 首次空 msglist，刷新 cookie 重试 (comment_post)...")
                        _invalidate_session_cache()
                        new_session = _ensure_session()
                        if not new_session:
                            return "该用户暂无可见说说"
                        cur_session = new_session
                        ctx = await cur_session.get_ctx()

                    feeds_resp = await client.get(list_url, params={
                        "g_tk": ctx.gtk2, "uin": uid, "ftype": 0, "sort": 0,
                        "pos": pos, "num": 7, "replynum": 100, "format": "json", "need_comment": 1,
                    }, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "referer": f"https://user.qzone.qq.com/{ctx.uin}",
                        "origin": "https://user.qzone.qq.com",
                        "Cookie": ctx.cookie_header,
                    })
                    parsed = QzoneParser.parse_response(feeds_resp.text)
                    if parsed.get("code") != 0:
                        continue
                    msglist = parsed.get("msglist") or []
                    if msglist:
                        break
                else:
                    return "该用户暂无可见说说"

            post = msglist[0]
            tid_to_use = post.get("tid", "")
            post_uin = post.get("uin", 0)
            if not tid_to_use:
                return "说说 ID 为空"

        # 评论
        async def _comment():
            api = QzoneAPI(cur_session)
            try:
                comment_resp = await api.comment(post_uin, tid_to_use, content)
                if comment_resp.ok:
                    return f"已评论 {post.get('name', '未知用户')} 的说说：「{content}」"
                return f"评论失败: {comment_resp.message}"
            finally:
                await api.close()
        return await _comment()

    return _run_async(_do())


@tool
def qzone_lookup_member(name: str) -> str:
    """
    查找群成员的 QQ 号。当你不知道对方 QQ 号但知道群昵称或QQ昵称时使用。
    支持按角色查找：传 name="群主" 或 name="管理员"。
    查到 QQ 号后可以配合其他工具使用（如 qzone_view_feeds、qzone_comment_post 等）。

    Args:
        name: 对方的群昵称、QQ昵称，或"群主"/"管理员"
    """
    if not name:
        return "请告诉我你要找谁"

    uid = _resolve_user_id("", name)
    if not uid:
        return f"在群成员中没找到「{name}」，请确认昵称是否正确"

    # 获取详细信息
    try:
        resp = httpx.get(
            f"{NAPCAT_BASE_URL}/get_group_member_list",
            headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
            timeout=5,
        )
        groups = resp.json().get("data", [])
        for g in groups:
            mr = httpx.get(
                f"{NAPCAT_BASE_URL}/get_group_member_info",
                params={"group_id": g["group_id"], "user_id": uid},
                headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
                timeout=5,
            )
            info = mr.json().get("data", {})
            if info:
                card = info.get("card") or info.get("nickname", "")
                role_map = {"owner": "群主", "admin": "管理员", "member": "群成员"}
                role = role_map.get(info.get("role", ""), "群成员")
                return f"找到 {card}：QQ={uid}，身份={role}"
    except Exception:
        pass

    return f"找到 QQ={uid}"


@tool
def qzone_publish_post(text: str) -> str:
    """
    发布一条新说说到自己的 QQ 空间。

    Args:
        text: 说说正文内容
    """
    session = _ensure_session()
    if not session:
        return "QQ 空间功能未配置"

    async def _do():
        api = QzoneAPI(session)
        resp = await api.publish(text)
        await api.close()
        if resp.ok:
            return f"说说已发布：{text[:60]}"
        return f"发布失败: {resp.message}"

    return _run_async(_do())


@tool
def qzone_delete_post(tid: str, feeds_type: int = 1) -> str:
    """
    删除自己空间的一条说说。

    Args:
        tid: 说说的 tid（从 qzone_view_feeds 或 qzone_search_post 的结果中获取）
        feeds_type: 说说类型（从 qzone_view_feeds 的结果中获取，通常是 1 或 100）
    """
    session = _ensure_session()
    if not session:
        return "QQ 空间功能未配置"

    async def _do():
        api = QzoneAPI(session)
        resp = await api.delete_post(tid, feeds_type)
        await api.close()
        if resp.ok:
            return f"说说 {tid} 已删除"
        return f"删除失败: {resp.message}"

    return _run_async(_do())


@tool
def qzone_view_visitor() -> str:
    """查看自己 QQ 空间的最近访客"""
    session = _ensure_session()
    if not session:
        return "QQ 空间功能未配置"

    async def _do():
        api = QzoneAPI(session)
        resp = await api.get_visitor()
        await api.close()
        if not resp.ok:
            return f"获取访客失败: {resp.message}"
        data = resp.data.get("data") or {}
        items = data.get("items") or []
        if not items:
            return "暂无访客记录"
        lines = ["最近访客:"]
        for v in items[:20]:
            name = v.get("name", "匿名")
            src_map = {0: "空间", 13: "动态", 32: "手机QQ", 41: "TIM"}
            src = src_map.get(v.get("src"), "未知")
            lines.append(f"  {name} 来自{src}")
        return "\n".join(lines)

    return _run_async(_do())


# ==================== 随机说说评论（后台任务） ====================

async def random_feed_comment(llm) -> None:
    """（后台任务）随机选一个群友，读取其最新说说并用 LLM 生成评论"""
    import random

    session = _ensure_session()
    if not session:
        return

    try:
        ctx = await session.get_ctx()
        bot_uin = str(ctx.uin)
    except Exception as e:
        print(f"[随机说说] 获取 session 失败: {e}")
        return

    # 获取所有群
    try:
        resp = httpx.get(
            f"{NAPCAT_BASE_URL}/get_group_list",
            headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
            timeout=5,
        )
        groups = resp.json().get("data", [])
        if not groups:
            print("[随机说说] 无可用群")
            return
    except Exception as e:
        print(f"[随机说说] 获取群列表失败: {e}")
        return

    # 收集所有群成员（排除自己）
    all_users: list[dict] = []
    for group in groups:
        try:
            mr = httpx.get(
                f"{NAPCAT_BASE_URL}/get_group_member_list",
                params={"group_id": group["group_id"]},
                headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
                timeout=5,
            )
            for m in mr.json().get("data", []):
                uid = str(m.get("user_id", ""))
                if uid and uid != bot_uin:
                    all_users.append({
                        "uid": uid,
                        "name": m.get("card") or m.get("nickname", ""),
                    })
        except Exception:
            continue

    if not all_users:
        print("[随机说说] 无可选用户")
        return

    # 随机选用户最多试 3 次
    list_url = "https://user.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msglist_v6"
    random.shuffle(all_users)

    for target in all_users[:5]:
        uid = target["uid"]
        name = target["name"]
        try:
            cur_session = session
            cur_ctx = ctx
            for attempt in range(2):
                if attempt == 1:
                    _invalidate_session_cache()
                    new_session = _ensure_session()
                    if not new_session:
                        break
                    cur_session = new_session
                    cur_ctx = await cur_session.get_ctx()

                async with httpx.AsyncClient() as client:
                    feeds_resp = await client.get(list_url, params={
                        "g_tk": cur_ctx.gtk2, "uin": uid, "ftype": 0, "sort": 0,
                        "pos": 0, "num": 1, "replynum": 100,
                        "format": "json", "need_comment": 1,
                    }, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "referer": f"https://user.qzone.qq.com/{cur_ctx.uin}",
                        "origin": "https://user.qzone.qq.com",
                        "Cookie": cur_ctx.cookie_header,
                    })
                    parsed = QzoneParser.parse_response(feeds_resp.text)
                    if parsed.get("code") != 0:
                        continue
                    msglist = parsed.get("msglist") or []
                    if not msglist:
                        continue

                    post = QzoneParser.parse_feeds([msglist[0]])[0]
                    post_text = post.get("text", "")
                    post_tid = post.get("tid", "")
                    post_uin = post.get("uin", 0)

                    if not post_tid or not post_text:
                        continue

                    # LLM 生成评论
                    prompt = (
                        f"你正在浏览QQ空间，随机刷到一条说说：\n"
                        f"{name}：{post_text}\n\n"
                        f"请用一句话评论这条说说（10-30字），语气自然友好。"
                    )
                    try:
                        resp = await llm.ainvoke(prompt)
                        comment = resp.content.strip().strip("\"'").strip()
                    except Exception as e:
                        print(f"[随机说说] LLM 生成评论失败: {e}")
                        continue

                    if not comment:
                        continue

                    # 发表评论
                    api = QzoneAPI(cur_session)
                    try:
                        comment_resp = await api.comment(int(post_uin), post_tid, comment)
                        if comment_resp.ok:
                            print(f"[随机说说] 已评论 {name}({uid}): 「{comment[:40]}」")
                        else:
                            print(f"[随机说说] 评论失败 {name}: {comment_resp.message}")
                    finally:
                        await api.close()
                    return  # 成功就退出

        except Exception as e:
            print(f"[随机说说] 处理用户 {name}({uid}) 失败: {e}")
            continue
