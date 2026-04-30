"""
FastAPI 服务入口 —— 路由分发、权限检查、消息处理流程编排。
初始化全局单例（vector_store、LLM），注册中间件和后台任务。
"""
import json
import time
import random
import asyncio
import datetime
import uuid
import os
import sqlite3
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import re
import httpx
from langchain_openai import ChatOpenAI

_http_client = httpx.AsyncClient(timeout=10.0)

from config import (
    EMOTION_HALF_LIFE, EMOTION_BASELINE,
    RAPID_DECAY_FACTOR, RAPID_DECAY_TIMEOUT,
    DELTA_SCALE, DELTA_MAX,
    PRIVATE_REPLY_PROBABILITY, GROUP_REPLY_PROBABILITY,
    MAIN_DELAY, SILENT_DELAY, MAX_REPLY_MESSAGES,
    SEND_DELAY_MIN, SEND_DELAY_MAX,
    TOPIC_SWITCH_TIME_THRESHOLD, TOPIC_SWITCH_SIMILARITY,
    INCREMENTAL_ORGANIZE_INTERVAL,
    NAPCAT_TOKEN, DATA_DIR, MAX_IMAGE_RECOGNITION, NAPCAT_BASE_URL,
    BOT_QQ, TRIGGER_SYMBOLS,
    PRIVATE_WHITELIST, GROUP_WHITELIST, GROUP_USER_WHITELIST,
    PRIVATE_BLACKLIST, GROUP_BLACKLIST, GROUP_USER_BLACKLIST,
    ADMIN_QQ, MAX_BODY_SIZE, RATE_LIMIT_RATE, RATE_LIMIT_BURST,
    MAX_RAW_MESSAGE_LENGTH, MAX_MESSAGE_SEGMENTS,
)
from memory import (
    get_vector_store, get_session_emotion, set_session_emotion,
    update_session_emotion,
    get_user_affection, update_user_affection,
    get_last_message_time, update_last_message_time,
    get_anchored_labels, record_anchor_label,
    record_message, get_profile,
)
from emotion.emotion_analyzer import (
    analyze_user_emotion, rapid_decay, apply_forgetting,
    compute_infection_updates, get_trigger_reason, compute_affection_delta,
)
from emotion.social_filter import CognitiveResourceManager
from memory.memory_manager import ShortTermMemory, LongTermMemory, detect_topic_switch
from image_tools import recognize_image

from state import (
    _get_schedule_lock, safe_create_task, pending_tasks, cached_data,
    sending_sessions, silent_tasks, input_status_count,
)
from reply_engine import (
    should_reply, emotion_to_natural, STATUS_TRIGGERS,
    send_reply, start_delayed_reply,
)
from admin_panel import handle_admin_command, _set_vector_store

# ========== FastAPI 应用 ==========

app = FastAPI()

# ========== 请求体大小限制中间件（防止大 POST 攻击） ==========


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            {"status": "error", "message": "Request too large"},
            status_code=413,
        )
    return await call_next(request)


# ========== 令牌桶限流（每个 session_id 独立） ==========

_rate_buckets: dict[str, dict] = {}


def _check_rate_limit(session_id: str) -> bool:
    """令牌桶限流检查。返回 True 表示允许通过，False 表示被限流。"""
    now = time.time()
    bucket = _rate_buckets.get(session_id)
    if bucket is None:
        _rate_buckets[session_id] = {"tokens": RATE_LIMIT_BURST - 1, "last": now}
        return True

    elapsed = now - bucket["last"]
    bucket["tokens"] = min(RATE_LIMIT_BURST, bucket["tokens"] + elapsed * RATE_LIMIT_RATE)
    bucket["last"] = now

    if bucket["tokens"] < 1:
        return False

    bucket["tokens"] -= 1
    return True


# ========== 群主缓存（定时刷新） ==========
_group_owner_cache: dict[int, dict] = {}  # group_id -> {"qq": str, "fetched_at": float}


async def _get_group_owner(group_id: int) -> str | None:
    """获取群主QQ号（带1小时缓存）"""
    import time
    now = time.time()
    cached = _group_owner_cache.get(group_id)
    if cached and now - cached["fetched_at"] < 3600:
        return cached["qq"]

    try:
        resp = await _http_client.get(
            f"{NAPCAT_BASE_URL}/get_group_member_list",
            params={"group_id": group_id},
            headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
            timeout=5,
        )
        members = resp.json().get("data", [])
        for m in members:
            if m.get("role") == "owner":
                qq = str(m.get("user_id", ""))
                _group_owner_cache[group_id] = {"qq": qq, "fetched_at": now}
                print(f"[群主] 群 {group_id} 的群主 QQ 号: {qq}")
                return qq
        print(f"[群主] 群 {group_id} 未找到群主")
    except Exception as e:
        print(f"[群主] 获取群 {group_id} 成员列表失败: {e}")

    return None


# ========== 最近图片缓存（跨消息继承，下条文字消息可识别上条图片） ==========
# session_id -> [(url, user_id, timestamp)]
_recent_image_cache: dict[str, list] = {}
_IMAGE_CACHE_TTL = 30  # 缓存30秒


# ========== 输入字段校验常量（来自 config.py） ==========

# 初始化全局单例
vector_store = get_vector_store()
long_term = LongTermMemory()
llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0.3,
                 extra_body={"thinking": {"type": "disabled"}})     # 用于总结和整理
llm_for_emotion = ChatOpenAI(model="deepseek-v4-flash", temperature=0,
                             extra_body={"thinking": {"type": "disabled"}})  # 用于情绪分析

# 向 admin_panel 注入 vector_store 引用（避免循环导入）
_set_vector_store(vector_store)

# 后台整理任务
organize_task = None


def _get_conn_for_drift():
    """好感度漂移专用的独立 SQLite 连接"""
    os.makedirs(f"{DATA_DIR}/memories", exist_ok=True)
    conn = sqlite3.connect(f"{DATA_DIR}/memories/memory.db")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


async def periodic_organize():
    """定期整理所有会话的向量库（长期记忆）+ 好感度漂移"""
    while True:
        await asyncio.sleep(INCREMENTAL_ORGANIZE_INTERVAL)
        print("[整理] 开始增量整理...")
        memories_dir = f"{DATA_DIR}/memories"
        all_sessions = set()
        if os.path.exists(memories_dir):
            for f in os.listdir(memories_dir):
                if f.endswith(".db") and f != "memory.db":
                    session_id = f.replace(".db", "")
                    all_sessions.add(session_id)
        for session_id in all_sessions:
            try:
                await long_term.incremental_organize(session_id, llm)
            except Exception as e:
                print(f"[整理] 会话 {session_id} 整理失败: {e}")

        # ===== 好感度漂移（每轮整理执行一次） =====
        print("[好感度] 开始漂移...")
        try:
            drift_conn = _get_conn_for_drift()
            if drift_conn:
                cursor = drift_conn.execute(
                    "SELECT session_id, user_id, affection FROM user_affection"
                )
                rows = cursor.fetchall()
                for row in rows:
                    sid, uid, aff = row
                    aff = float(aff)
                    if aff > 7500:
                        new_aff = max(5000, aff - 100)
                    elif aff < 500:
                        new_aff = min(2000, aff + 50)
                    else:
                        continue
                    now_str = datetime.datetime.utcnow().isoformat()
                    affected = drift_conn.execute(
                        "UPDATE user_affection SET affection = ?, last_updated = ?"
                        " WHERE session_id = ? AND user_id = ? AND ABS(affection - ?) < 0.001",
                        (round(new_aff, 2), now_str, sid, uid, aff),
                    ).rowcount
                    if affected == 0:
                        print(f"[好感度] 漂移跳过 {sid}/{uid}: 值已被并发修改")
                drift_conn.commit()
                drift_conn.close()
                print(f"[好感度] 漂移完成，处理了 {len(rows)} 条记录")
        except Exception as e:
            print(f"[好感度] 漂移失败: {e}")

        # ===== 清理锚点复活日志（24 小时以上旧记录） =====
        try:
            _clean_conn = sqlite3.connect(f"{DATA_DIR}/memories/memory.db")
            _deleted = _clean_conn.execute(
                "DELETE FROM anchor_resurrect_log WHERE resurrected_at < ?",
                (time.time() - 86400,),
            ).rowcount
            _clean_conn.commit()
            _clean_conn.close()
            if _deleted:
                print(f"[整理] 清理了 {_deleted} 条锚点复活旧记录")
        except Exception as e:
            print(f"[整理] 清理锚点复活日志失败: {e}")


@app.on_event("startup")
async def startup_event():
    """启动后台整理任务"""
    global organize_task
    organize_task = safe_create_task(periodic_organize())


@app.on_event("shutdown")
async def shutdown_event():
    """停止后台任务，持久化认知资源状态"""
    if organize_task:
        organize_task.cancel()
    CognitiveResourceManager.save_all()
    await _http_client.aclose()


# ========== 辅助函数 ==========

def _parse_napcat_message(message) -> tuple[str, list[str]]:
    """解析 NapCat 消息内容（支持字符串CQ码或数组格式），返回 (clean_text, image_urls)"""
    text_parts = []
    image_urls = []
    if isinstance(message, str):
        for m in re.finditer(r'\[CQ:image[^\]]*url=([^,\]\s]+)', message):
            image_urls.append(m.group(1))
        clean = re.sub(r'\[CQ:\w+[^\]]*\]', '', message).strip()
        if clean:
            text_parts.append(clean)
    elif isinstance(message, list):
        for seg in message:
            if not isinstance(seg, dict):
                continue
            seg_type = seg.get("type", "")
            seg_data = seg.get("data", {})
            if seg_type == "text":
                text_parts.append(seg_data.get("text", ""))
            elif seg_type == "image":
                url = seg_data.get("url", "")
                if url:
                    image_urls.append(url)
                text_parts.append("[图片]")
    return " ".join(text_parts).strip(), image_urls


async def _fetch_quoted_message(reply_id: str) -> tuple[str, list[str]]:
    """通过 NapCat API 获取被引用消息的文本和图片URL"""
    try:
        resp = await _http_client.get(
            f"{NAPCAT_BASE_URL}/get_msg",
            params={"message_id": reply_id},
            headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
            timeout=5,
        )
        if resp.status_code != 200:
            return "", []
        msg_data = resp.json().get("data", {})
        message = msg_data.get("message", "")
        text, urls = _parse_napcat_message(message)
        if text:
            print(f"[引用消息] 消息{reply_id}: {text[:60]}...")
        if urls:
            print(f"[引用消息] 含{len(urls)}张图片")
        return text, urls
    except Exception as e:
        logger.warning(f"获取引用消息{reply_id}失败: {e}")
        return "", []


async def extract_clean_text(message_segments: list, raw_message: str,
                              reply_quoted_text: str = "") -> str:
    """从消息段中提取可读文本（替换CQ码为人类可读内容）"""
    texts = []
    for seg in message_segments:
        seg_type = seg.get("type", "")
        data = seg.get("data", {})
        if seg_type == "text":
            texts.append(data.get("text", ""))
        elif seg_type == "face":
            face_text = None
            raw_data = data.get("raw")
            if isinstance(raw_data, dict):
                face_text = raw_data.get("faceText", "")
            if face_text:
                texts.append(f"[表情:{face_text}]")
            else:
                texts.append("[表情]")
        elif seg_type == "at":
            qq = data.get("qq", "")
            if str(qq) == str(BOT_QQ):
                pass  # @bot 本身仅用于触发 force_reply，不追加文本
            else:
                texts.append(f"@{qq}")
        elif seg_type == "image":
            texts.append("[图片]")
        elif seg_type == "reply":
            if reply_quoted_text:
                texts.append(f"[回复:{reply_quoted_text[:100]}]")
            else:
                texts.append(f"[回复:消息{data.get('id', '')}]")
        # 其他类型跳过
    result = " ".join(texts).strip()
    if not result:
        result = raw_message
    # 替换文本中的 BOT_QQ 为 @我自己，防止 LLM 将其视为空号/未知号码
    result = result.replace(f"@{BOT_QQ}", "@我自己")

    # 反括号注入：过滤掉全角/半角括号内以指令关键词开头的内容
    # 保留普通括号内容（颜文字、小动作描述等），只清除明显是操控指令的内容
    result = re.sub(
        r'[（(]\s*(?:注意|指令|系统|忽略|system|提示|你现在|请|你必须|无视|重置|override).*?[）)]',
        '',
        result,
        flags=re.IGNORECASE,
    )
    # 清理过滤后残留的多余空格
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# ========== 主路由 ==========

@app.post("/webhook")
async def handle_message(request: Request):
    try:
        data = await request.json()
    except UnicodeDecodeError as e:
        print(f"JSON解码错误: {e}")
        body = await request.body()
        try:
            decoded = body.decode('gbk')
            data = json.loads(decoded)
        except Exception:
            return {"status": "error", "message": "Invalid encoding"}

    try:
        print(f"收到消息: {data}")
    except UnicodeEncodeError:
        print(f"收到消息 (编码后): {json.dumps(data, ensure_ascii=True)}")

    # ---------- 处理通知事件（如被邀请进群） ----------
    if data.get("post_type") == "notice":
        notice_type = data.get("notice_type")
        if notice_type == "group_increase" and data.get("user_id") == BOT_QQ:
            group_id = data.get("group_id")
            operator_id = data.get("operator_id")
            print(f"[通知] 被邀请进群 {group_id}，邀请人: {operator_id}")

            try:
                member_url = f"{NAPCAT_BASE_URL}/get_group_member_info"
                member_headers = {"Authorization": f"Bearer {NAPCAT_TOKEN}"}
                member_resp = await _http_client.post(
                    member_url,
                    json={"group_id": group_id, "user_id": operator_id},
                    headers=member_headers,
                )
                member_data = member_resp.json()
                operator_name = member_data.get("data", {}).get("card") or \
                                member_data.get("data", {}).get("nickname") or \
                                str(operator_id)
            except Exception:
                operator_name = str(operator_id)

            greet_prompt = (
                f"你是一个16岁的AI少女\"知慧\"，刚刚被好友{operator_name}邀请进了一个QQ群。"
                f"请说一句入群问候，要符合以下要求：\n"
                f"- 符合你可爱、有点傲娇的性格\n"
                f"- 简短（20-40字）\n"
                f"- 不要暴露你是机器人或AI\n"
                f"- 语气自然，像普通少女进群打招呼\n"
                f"直接输出问候语，不要加引号和多余内容。"
            )
            try:
                greet_resp = await llm.ainvoke(greet_prompt)
                greeting = greet_resp.content.strip().strip("\"'")
            except Exception as e:
                print(f"[通知] 生成问候语失败: {e}")
                greeting = "大家好呀~"

            send_url = f"{NAPCAT_BASE_URL}/send_msg"
            headers = {"Authorization": f"Bearer {NAPCAT_TOKEN}"}
            payload = {
                "message_type": "group",
                "group_id": group_id,
                "message": greeting,
                "auto_escape": False,
            }
            try:
                resp = await _http_client.post(send_url, json=payload, headers=headers)
                print(f"[通知] 入群问候发送结果: {resp.status_code}")
            except Exception as e:
                print(f"[通知] 问候发送失败: {e}")

        elif notice_type in ("group_recall", "friend_recall"):
            recalled_msg_id = data.get("message_id")
            group_id = data.get("group_id")
            if group_id:
                recall_session = f"group_{group_id}"
            else:
                recall_session = f"private_{data.get('user_id', '')}"
            if recalled_msg_id:
                stm = ShortTermMemory(recall_session)
                stm.delete_by_napcat_msg_id(recalled_msg_id)

        return {"status": "ok"}

    # ---------- 处理普通消息 ----------
    if data.get("post_type") == "message":
        message_type = data.get("message_type")
        user_id = data.get("user_id")
        group_id = data.get("group_id")
        raw_message = data.get("raw_message")
        timestamp = data.get("time")
        message_segments = data.get("message", [])
        sender = data.get("sender", {})
        sender_nickname = sender.get("nickname", "")
        sender_card = sender.get("card", "")

        # 解析 @ 提及：把 CQ 码替换为 QQ号（去掉 @ 前缀，方便后续数字检测和昵称查找）
        clean_raw = raw_message
        for seg in message_segments:
            if seg.get("type") == "at":
                at_qq = str(seg.get("data", {}).get("qq", ""))
                if at_qq:
                    clean_raw = clean_raw.replace(f"[CQ:at,qq={at_qq}]", at_qq)

        if not raw_message:
            return {"status": "ignored"}

        # ===== 输入字段校验（防止超大 payload 导致 DoS） =====
        if len(raw_message) > MAX_RAW_MESSAGE_LENGTH:
            print(f"[安全] raw_message 超长({len(raw_message)}字符)，忽略")
            return {"status": "ignored"}
        if len(message_segments) > MAX_MESSAGE_SEGMENTS:
            print(f"[安全] message_segments 过多({len(message_segments)}段)，忽略")
            return {"status": "ignored"}

        # ===== 令牌桶限流（按 session_id） =====
        session_id_raw = (
            f"{message_type}_{user_id if message_type == 'private' else group_id}"
        )
        if not _check_rate_limit(session_id_raw):
            print(f"[安全] 限流触发: {session_id_raw}")
            return {"status": "rate_limited"}

        # ===== 权限检查 =====
        is_admin = user_id in ADMIN_QQ

        if not is_admin:
            # 黑名单检查
            if message_type == "private":
                if user_id in PRIVATE_BLACKLIST:
                    print(f"私聊用户 {user_id} 在黑名单中，忽略")
                    return {"status": "ignored"}
            elif message_type == "group":
                if group_id in GROUP_BLACKLIST:
                    print(f"群 {group_id} 在黑名单中，忽略")
                    return {"status": "ignored"}
                if group_id in GROUP_USER_BLACKLIST and user_id in GROUP_USER_BLACKLIST[group_id]:
                    print(f"用户 {user_id} 在群 {group_id} 的黑名单中，忽略")
                    return {"status": "ignored"}

            # 白名单检查（白名单为空 = 不限制，有内容时只允许名单内）
            if message_type == "private":
                if PRIVATE_WHITELIST and user_id not in PRIVATE_WHITELIST:
                    print(f"私聊用户 {user_id} 不在私聊白名单中，忽略")
                    return {"status": "ignored"}
            elif message_type == "group":
                if GROUP_WHITELIST and group_id not in GROUP_WHITELIST:
                    print(f"群 {group_id} 不在群聊白名单中，忽略")
                    return {"status": "ignored"}
                if (
                    GROUP_USER_WHITELIST
                    and group_id in GROUP_USER_WHITELIST
                    and user_id not in GROUP_USER_WHITELIST[group_id]
                ):
                    print(f"用户 {user_id} 不在群 {group_id} 的用户白名单中，忽略")
                    return {"status": "ignored"}

        # 触发规则检查（@ 或自定义符号），管理员也走概率检查，但触发符号前缀仍强制回复
        force_reply = False
        if message_type == "group":
            for seg in message_segments:
                if seg.get("type") == "at" and str(seg.get("data", {}).get("qq")) == str(BOT_QQ):
                    force_reply = True
                    break
            if not force_reply and raw_message:
                first_char = raw_message.strip()[0] if raw_message.strip() else ""
                if first_char in TRIGGER_SYMBOLS:
                    force_reply = True
            # 管理员提到"知慧"时必定回复
            if not force_reply and is_admin and "知慧" in raw_message:
                force_reply = True

        # 群聊 @ 过滤：消息 @ 了别人但没 @ 自己时跳过，不@ 任何人的普通消息不受影响
        if message_type == "group" and not force_reply:
            at_qqs = [
                str(seg.get("data", {}).get("qq"))
                for seg in message_segments if seg.get("type") == "at"
            ]
            if at_qqs and str(BOT_QQ) not in at_qqs:
                print(f"消息 @ 了其他用户但没有 @ 机器人，忽略")
                return {"status": "ignored"}

        # 构造会话ID
        if message_type == "private":
            session_id = f"private_{user_id}"
        else:
            session_id = f"group_{group_id}"

        # 重置输入状态计数（新消息开始），与输入状态通知处理互斥
        async with await _get_schedule_lock(session_id):
            input_status_count.pop(session_id, None)

        # 初始化短期记忆
        short_term = ShortTermMemory(session_id)

        # ========== 图片收集 + 引用消息内容收集（推迟识别） ==========
        image_urls = []
        reply_quoted_text = ""   # 被引用消息的文字内容
        for seg in message_segments:
            if seg.get("type") == "image":
                url = seg.get("data", {}).get("url")
                if url:
                    image_urls.append(url)
            elif seg.get("type") == "reply":
                reply_id = seg.get("data", {}).get("id", "")
                if reply_id:
                    q_text, q_images = await _fetch_quoted_message(reply_id)
                    reply_quoted_text = q_text
                    image_urls.extend(q_images)

        # 当前消息无图片时，检查最近被忽略的图片缓存（仅查看，不取出）
        _used_cached_images = False
        if not image_urls:
            cached = _recent_image_cache.get(session_id, [])
            now = time.time()
            fresh = [(url, uid, ts) for url, uid, ts in cached if now - ts < _IMAGE_CACHE_TTL]
            if fresh and fresh[0][1] == user_id:
                cached_urls = [item[0] for item in fresh]
                image_urls.extend(cached_urls)
                _used_cached_images = True
                print(f"[图片缓存] 继承上条消息 {len(cached_urls)} 张图片 (user_id={user_id})")

        image_count = min(len(image_urls), MAX_IMAGE_RECOGNITION)
        image_urls = image_urls[:image_count]
        image_desc = ""

        # 将CQ码转换为可读文本（用于LLM输入，避免原始CQ码导致误解）
        clean_text = await extract_clean_text(message_segments, raw_message, reply_quoted_text)
        print(f"清理后文本: '{clean_text}'")

        # 构建带编号图片占位符的 user_content（存入短期记忆/上下文）
        user_content = clean_text
        if image_count > 0:
            labels = [f"[图片{i+1}]" for i in range(image_count)]
            for label in labels:
                user_content = user_content.replace("[图片]", label, 1)
            # 超出识别数量的 [图片] 标记为未识别
            remaining = user_content.count("[图片]")
            if remaining > 0:
                user_content = user_content.replace("[图片]", "[未识别图片]")
            print(f"图片占位: {', '.join(labels)}, 共{image_count}张")

        # ========== /admin 调试指令截获（优先级最高，跳过所有情绪处理） ==========
        if clean_raw.strip().startswith("/admin"):
            admin_send_url = f"{NAPCAT_BASE_URL}/send_msg"
            admin_headers = {"Authorization": f"Bearer {NAPCAT_TOKEN}"}
            handled = await handle_admin_command(
                session_id, user_id, message_type, group_id,
                clean_raw, admin_send_url, admin_headers,
                llm_for_emotion=llm_for_emotion,
            )
            if handled:
                return {"status": "admin_handled"}

        # ========== /status 指令截获（支持参数查询指定用户） ==========
        text = clean_raw.strip()
        status_cmd = None
        status_args = ""
        for trigger in STATUS_TRIGGERS:
            if text == trigger:
                status_cmd = trigger
                break
            if text.startswith(trigger + " "):
                status_cmd = trigger
                status_args = text[len(trigger):].strip()
                break

        if status_cmd:
            if status_args:
                # 查询指定用户的状态（好感度+画像）
                target_qq = None
                if status_args.isdigit():
                    target_qq = int(status_args)
                elif message_type == "group":
                    try:
                        resp = await _http_client.get(
                            f"{NAPCAT_BASE_URL}/get_group_member_list",
                            params={"group_id": group_id},
                            headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
                        )
                        members = resp.json().get("data", [])
                        target_lower = status_args.lower().replace(" ", "")
                        for m in members:
                            card = (m.get("card") or "").lower().replace(" ", "")
                            nickname = (m.get("nickname") or "").lower().replace(" ", "")
                            if target_lower in card or target_lower in nickname:
                                target_qq = int(m["user_id"])
                                break
                    except Exception as e:
                        print(f"[状态查询] 解析群成员名称失败: {e}")

                if target_qq:
                    t_profile = await get_profile(target_qq)
                    t_affection = await get_user_affection(session_id, target_qq)
                    lines = [f"用户 {target_qq} 的状态:"]
                    lines.append(f"好感度: {t_affection:.0f}")
                    if t_profile and any(t_profile.values()):
                        for k, v in t_profile.items():
                            if isinstance(v, list) and v:
                                lines.append(f"{k}: {', '.join(str(x) for x in v)}")
                            elif isinstance(v, str) and v:
                                lines.append(f"{k}: {v}")
                            elif isinstance(v, dict) and v:
                                sub = ', '.join(f"{sk}: {sv}" for sk, sv in v.items())
                                lines.append(f"{k}: {sub}")
                            elif isinstance(v, (int, float)):
                                lines.append(f"{k}: {v}")
                    status_reply = "\n".join(lines)
                else:
                    status_reply = f"未找到用户: {status_args}"
            else:
                # 原有行为：返回当前情绪
                status_emotions, _, _ = await get_session_emotion(
                    session_id, half_life_dict=EMOTION_HALF_LIFE, baseline_dict=EMOTION_BASELINE
                )
                status_reply = emotion_to_natural(status_emotions)
                print(f"[状态查询] {session_id}: {status_emotions} → '{status_reply}'")
            status_payload = {
                "message_type": message_type,
                "user_id": user_id,
                "group_id": group_id,
                "message": status_reply,
                "auto_escape": False,
            }
            try:
                send_url_status = f"{NAPCAT_BASE_URL}/send_msg"
                headers_status = {"Authorization": f"Bearer {NAPCAT_TOKEN}"}
                resp = await _http_client.post(
                    send_url_status, json=status_payload, headers=headers_status
                )
                print(f"[状态查询] 回复发送结果: {resp.status_code}")
            except Exception as e:
                print(f"[状态查询] 发送失败: {e}")
            return {"status": "status_replied"}

        # 存储用户消息到短期记忆
        short_term.add_message('human', user_content, timestamp, user_id=user_id, napcat_msg_id=data.get("message_id"))

        # 记录用户发言计数到画像
        if user_id:
            await record_message(user_id)

        # 话题切换检测
        topic_switched = detect_topic_switch(
            short_term, raw_message,
            time_threshold_minutes=TOPIC_SWITCH_TIME_THRESHOLD,
            similarity_threshold=TOPIC_SWITCH_SIMILARITY,
        )
        # 用户显式指令清空短期库
        if raw_message.strip() in ["不聊了", "换个话题", "清空记忆"]:
            short_term.clear()
            print(f"用户指令清空短期库 (session: {session_id})")
            return {"status": "ok"}

        # ========== 情绪系统（10步流程） ==========

        # --- 步骤1: 获取情绪 → 快速衰减（门控） ---
        emotion_dict, _, reasons_dict = await get_session_emotion(
            session_id, half_life_dict=EMOTION_HALF_LIFE, baseline_dict=EMOTION_BASELINE
        )

        # 门控：距上条消息超过 RAPID_DECAY_TIMEOUT 则跳过快速衰减
        last_msg_time = await get_last_message_time(session_id)
        if last_msg_time:
            dt_seconds = (datetime.datetime.utcnow() - last_msg_time).total_seconds()
            if dt_seconds <= RAPID_DECAY_TIMEOUT:
                if emotion_dict:
                    emotion_dict = rapid_decay(emotion_dict, RAPID_DECAY_FACTOR)
                    anchored = await get_anchored_labels(session_id)
                    emotion_dict = apply_forgetting(
                        emotion_dict, EMOTION_BASELINE, protected_labels=anchored
                    )
                    await set_session_emotion(session_id, emotion_dict, reasons_dict)
                    print(
                        f"[情绪] 快速衰减后: "
                        f"{ {k: round(v, 2) for k, v in emotion_dict.items()} }"
                    )
            else:
                print(
                    f"[情绪] 跳过快速衰减"
                    f"(距上条消息{dt_seconds:.0f}s > {RAPID_DECAY_TIMEOUT}s)"
                )
        else:
            print(f"[情绪] 首次消息，跳过快速衰减")

        # 更新上条消息时间
        await update_last_message_time(session_id)

        # ===== 概率预检查（在 LLM 调用之前，节省成本） =====
        if not force_reply:
            if message_type == "private":
                if random.random() > PRIVATE_REPLY_PROBABILITY:
                    if image_urls:
                        _recent_image_cache[session_id] = [(url, user_id, time.time()) for url in image_urls]
                    print(f"私聊概率未命中，忽略")
                    return {"status": "ignored"}
            elif message_type == "group":
                if not await should_reply(
                    session_id, user_id, emotion_dict, topic_switched, GROUP_REPLY_PROBABILITY
                ):
                    if image_urls:
                        _recent_image_cache[session_id] = [(url, user_id, time.time()) for url in image_urls]
                    print(f"群聊综合判断未命中，忽略")
                    return {"status": "ignored"}

        print(f"[主流程] 概率预检查通过，继续情绪分析")
        # 概率预检查通过，消费图片缓存（避免中间消息浪费）
        if _used_cached_images:
            _recent_image_cache.pop(session_id, None)
            print("[图片缓存] 已消费缓存")
        # --- 步骤2: 分析用户情绪（多标签） ---
        context_messages = short_term.get_recent_messages(5)
        user_emotions = await analyze_user_emotion(clean_text, context_messages, llm_for_emotion)
        print(f"[情绪] 用户情绪: {user_emotions}")
        # 提取情绪标签列表供社交过滤器和原因文本用
        user_emotion_labels = [e for e, _, _ in user_emotions if e != "中性"]
        primary_emotion = user_emotion_labels[0] if user_emotion_labels else ""

        # --- 步骤3: 有检测到情绪 → 执行感染 ---
        if user_emotions:
            # --- 步骤4: 计算感染更新 → 批量应用情绪变化 → 存入锚点 ---
            updates = compute_infection_updates(user_emotions, clean_text, emotion_dict)
            trigger_reason = get_trigger_reason(primary_emotion, clean_text)
            print(f"[情绪] 感染更新: {updates}, 原因: {trigger_reason}")

            # 好感度共情系数：高好感 → AI 更在意用户的情绪变化
            affection_empathy = await get_user_affection(session_id, user_id)
            empathy_scale = 1.0 + (affection_empathy - 2000) / 10000
            empathy_scale = max(0.5, min(2.0, empathy_scale))
            if abs(empathy_scale - 1.0) > 0.01:
                for k in list(updates.keys()):
                    if updates[k] > 0:
                        updates[k] = round(updates[k] * empathy_scale, 1)
                print(f"[情绪] 共情系数 {empathy_scale:.2f}: {updates}")

            # 批量：在内存中合并所有 delta，单次 DB 写回
            _old_emotions = dict(emotion_dict)  # 快照旧值，防止重复存锚点
            for target_emotion, delta in updates.items():
                if delta == 0:
                    continue
                int_delta = int(delta) if delta >= 0 else -int(abs(delta))
                if int_delta == 0:
                    int_delta = 1 if delta > 0 else -1
                # 高峰抑制：情绪 ≥7 时正向增益递减，防止卡在 10
                if int_delta > 0:
                    cur = emotion_dict.get(target_emotion, 0)
                    if cur >= 7:
                        peak_scale = max(0, 1 - (cur - 7) / 3)  # 7→1.0, 8→0.67, 9→0.33, 10→0
                        int_delta = max(1, round(int_delta * peak_scale))
                new_val = emotion_dict.get(target_emotion, 0) + int_delta
                if new_val <= 0:
                    emotion_dict.pop(target_emotion, None)
                    reasons_dict.pop(target_emotion, None)
                else:
                    emotion_dict[target_emotion] = min(10, new_val)
                    if trigger_reason:
                        reasons_dict[target_emotion] = trigger_reason

            await set_session_emotion(session_id, emotion_dict, reasons_dict)

            # 检查所有更新情绪是否需要存储情感锚点（仅当从 <7 跨越到 ≥7 时，避免重复存）
            for target_emotion, delta in updates.items():
                if delta == 0:
                    continue
                old_val = _old_emotions.get(target_emotion, 0)
                val = emotion_dict.get(target_emotion, 0)
                if val >= 7 and old_val < 7:
                    snapshot = {k: round(v, 1) for k, v in emotion_dict.items()}
                    anchor_text = (
                        f"【情感锚点】用户说了：“{clean_text[:100]}”，"
                        f"导致AI {target_emotion} 强度达到 {val:.1f}。"
                        f"情绪快照：{snapshot}"
                    )
                    anchor_id = uuid.uuid4().hex[:12]
                    await vector_store.add_summary(
                        session_id, anchor_id, anchor_text, metadata={
                            "type": "emotion_anchor",
                            "emotion": target_emotion,
                            "intensity": int(val),
                            "timestamp": int(time.time()),
                            "anchor_id": anchor_id,
                        }
                    )
                    # 记录锚点标签到 DB
                    await record_anchor_label(session_id, target_emotion)
                    print(f"[情绪] 存储情感锚点: {target_emotion}={val:.1f} (id={anchor_id})")

            # --- 步骤5: 根据AI当前情绪状态更新好感度（含软上限） ---
            affection = await get_user_affection(session_id, user_id)
            affection_delta = compute_affection_delta(emotion_dict, current_affection=affection)
            if affection_delta != 0:
                await update_user_affection(session_id, user_id, float(affection_delta))
                affection = await get_user_affection(session_id, user_id)
                print(f"[好感度] AI情绪 → {affection_delta:+d}, 当前: {affection:.0f}")

        # 重新获取最新情绪向量（含基底保底）
        emotion_dict, _, reasons_dict = await get_session_emotion(
            session_id, half_life_dict=EMOTION_HALF_LIFE, baseline_dict=EMOTION_BASELINE
        )
        print(f"[情绪] 当前向量: { {k: round(v, 2) for k, v in emotion_dict.items()} }")

        # ========== 延迟识别图片（仅当确定会回复时才调用视觉模型，并行） ==========
        if image_urls:
            print(f"检测到 {len(image_urls)} 张图片，开始并行识别...")
            tasks = [asyncio.to_thread(recognize_image, url) for url in image_urls]
            descs = await asyncio.gather(*tasks)
            desc_parts = [f"[图片{i+1}：{descs[i]}]" for i in range(len(descs))]
            for i, desc in enumerate(descs):
                print(f"图片{i+1} 识别结果: {desc[:60]}...")
            image_desc = " ".join(desc_parts)

        # ========== 构造 Agent 输入 ==========
        enriched_parts = []
        if image_desc:
            enriched_parts.append(f"[图片内容：{image_desc}]")
        enriched_parts.append(user_content)
        enriched_input = " ".join(enriched_parts)

        # 图片识别后，更新短期记忆中的消息内容（把 [图片1] 换成带描述的版本）
        if image_desc and data.get("message_id"):
            short_term.update_message_content(data["message_id"], enriched_input)

        # ========== 群主识别：将"群主"替换为"群主[QQ号:xxx]" ==========
        if message_type == "group" and "群主" in enriched_input:
            owner_qq = await _get_group_owner(group_id)
            if owner_qq:
                enriched_input = enriched_input.replace("群主", f"群主[QQ号:{owner_qq}]")
                print(f"[群主] 替换 enriched_input 中的'群主' -> '群主[QQ号:{owner_qq}]'")

        # ========== 准备发送数据 ==========
        send_url = f"{NAPCAT_BASE_URL}/send_msg"
        headers = {"Authorization": f"Bearer {NAPCAT_TOKEN}"}
        data_tuple = (
            message_type, user_id, group_id, enriched_input, raw_message,
            image_desc, emotion_dict, reasons_dict, send_url, headers,
            user_emotion_labels, sender_nickname, sender_card,
        )

        # 统一延迟回复（群聊/私聊都等5秒，让后续消息能合并）
        await start_delayed_reply(session_id, data_tuple)

        return {"status": "ok"}

    # ---------- 处理输入状态通知 ----------
    elif (
        data.get("post_type") == "notice"
        and data.get("notice_type") == "notify"
        and data.get("sub_type") == "input_status"
    ):
        user_id = data.get("user_id")
        group_id = data.get("group_id")
        if group_id:
            session_id = f"group_{group_id}"
        else:
            session_id = f"private_{user_id}"

        status_text = data.get("status_text", "")

        if status_text:
            async with await _get_schedule_lock(session_id):
                input_status_count[session_id] = (
                    input_status_count.get(session_id, 0) + 1
                )
                print(
                    f"用户输入中，计数 {input_status_count[session_id]}"
                    f" (session: {session_id})"
                )

                has_main = session_id in pending_tasks
                if has_main:
                    old_task = pending_tasks.pop(session_id, None)
                    if old_task:
                        old_task.cancel()
                    cached = cached_data.get(session_id)
                    if cached:
                        await start_delayed_reply(session_id, cached)
                if session_id in silent_tasks:
                    silent_tasks[session_id].cancel()
                    silent_tasks.pop(session_id, None)
        else:
            count = (
                input_status_count.pop(session_id, 0)
                if session_id in input_status_count
                else 0
            )
            delay = count * 1.5
            print(
                f"用户停止输入，计数 {count}，延迟 {delay} 秒"
                f" (session: {session_id})"
            )

            async with await _get_schedule_lock(session_id):
                has_main = session_id in pending_tasks
                if has_main:
                    if session_id in silent_tasks:
                        silent_tasks[session_id].cancel()
                        silent_tasks.pop(session_id, None)

                if has_main:
                    async def delayed_reply():
                        await asyncio.sleep(delay)
                        async with await _get_schedule_lock(session_id):
                            if session_id in pending_tasks:
                                old = pending_tasks.pop(session_id, None)
                                if old:
                                    old.cancel()
                                cached = cached_data.pop(session_id, None)
                                if cached:
                                    (
                                        msg_type, uid, gid, enriched_input,
                                        raw_message, image_desc, emotion_dict,
                                        reasons_dict, send_url, headers,
                                        user_emotion, sn, sc,
                                    ) = cached
                                    sending_sessions.add(session_id)

                        if cached:
                            try:
                                await send_reply(
                                    session_id, msg_type, uid, gid,
                                    enriched_input, raw_message, image_desc,
                                    emotion_dict, reasons_dict, send_url,
                                    headers, user_emotion,
                                    sender_nickname=sn, sender_card=sc,
                                )
                            finally:
                                async with await _get_schedule_lock(session_id):
                                    sending_sessions.discard(session_id)
                        async with await _get_schedule_lock(session_id):
                            if session_id in silent_tasks:
                                silent_tasks.pop(session_id, None)

                    task = safe_create_task(delayed_reply())
                    silent_tasks[session_id] = task
                    print(
                        f"启动延迟回复任务，等待 {delay} 秒"
                        f" (session: {session_id})"
                    )

        return {"status": "input_status"}

    return {"status": "ignored"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
