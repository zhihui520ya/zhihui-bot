"""
回复引擎 —— 回复决策、情绪转自然语言、消息分段、Agent 调用、延迟回复调度。
"""
import random
import time
import re
import asyncio
import uuid
import os
import datetime
import httpx
from langchain_core.messages import HumanMessage

# 共享异步 HTTP 客户端（连接复用）
_http_client = httpx.AsyncClient(timeout=10.0)
from memory import (
    get_vector_store, get_session_emotion, update_session_emotion,
    get_user_affection, get_anchored_labels, record_anchor_label,
    is_anchor_in_cooldown, record_anchor_resurrect,
    get_profile, extract_and_update,
)
from emotion.emotion_analyzer import (
    make_emotion_context, get_dominant_emotion, detect_emotion_shift,
    resurrect_from_anchor,
)
from emotion.social_filter import CognitiveResourceManager, apply_social_filter
from agent import invoke_agent
from music_tools import extract_music_card, strip_music_card_marker, CARD_TYPE_MAP
from qzone_tools import random_feed_comment
from langchain_openai import ChatOpenAI
from memory.memory_manager import ShortTermMemory
from config import (
    EMOTION_HALF_LIFE, EMOTION_BASELINE,
    MAIN_DELAY, MAX_REPLY_MESSAGES, SEND_DELAY_MIN, SEND_DELAY_MAX,
    NAPCAT_TOKEN,
)
from state import (
    _get_reply_lock, _get_schedule_lock, pending_tasks, cached_data,
    sending_sessions, silent_tasks, safe_create_task,
    ANCHOR_RESURRECT_COOLDOWN, _recently_resurrected,
)

# 用于总结的 LLM 实例（低 temperature，确保事实准确）
_llm_for_summary = ChatOpenAI(model="deepseek-v4-flash", temperature=0.3,
                               extra_body={"thinking": {"type": "disabled"}})

# ========== 情绪向量 → 自然语言描述 ==========

STATUS_TRIGGERS = {
    "/状态", "/status", "/status ",
    "知慧你现在怎么样", "知慧你现在怎样",
    "知慧你现在感觉怎么样", "知慧你现在感觉怎样",
}

EMOTION_DESCRIPTIONS = {
    "害羞": {
        (0, 0.5): "好像也没什么特别的感觉……",
        (0.5, 3): "有一点害羞……但是还好啦！",
        (3, 6): "有点害羞……不想被看到脸……（捂脸）",
        (6, 8): "我现在有点害羞……能不能别说这个了……",
        (8, 11): "太……太害羞了！/////",
    },
    "开心": {
        (0, 0.5): "心情还行吧……一般般。",
        (0.5, 2): "嗯…还行。",
        (2, 5): "心情还不错诶~",
        (5, 8): "挺开心的！虽然不想承认……",
        (8, 11): "超开心的！！（虽然不想承认）",
    },
    "生气": {
        (0, 0.5): "没生气。",
        (0.5, 2): "有点不爽……但还好。",
        (2, 5): "有点生气……别惹我。",
        (5, 8): "我正在生气。你确定要问我这个？",
        (8, 11): "非常生气。暂时不想说话。",
    },
    "委屈": {
        (0, 0.5): "没什么……",
        (0.5, 2): "有一点点难受……但不明显。",
        (2, 5): "有点委屈……（小声）",
        (5, 8): "委屈……不太想理你……（戳手指）",
        (8, 11): "好委屈……你不要和我说话……",
    },
    "醋意": {
        (0, 0.5): "谁会吃醋啊。",
        (0.5, 2): "有点在意……但也就一点点。",
        (2, 5): "哼……我才不在意呢。",
        (5, 8): "哦？我？我没吃醋啊。你继续。",
        (8, 11): "呵……你开心就好。（醋意爆棚）",
    },
    "撒娇": {
        (0, 0.5): "就很正常。",
        (0.5, 2): "想……想被人理一下……",
        (2, 5): "有点想你理我一下……",
        (5, 8): "陪陪我好不好嘛~",
        (8, 11): "唔……人家想要你陪我……（扯衣角）",
    },
}


def emotion_to_natural(emotions: dict) -> str:
    """
    将情绪向量转为自然语言描述。
    只描述强度 >= 1 的有效情绪，按主导情绪优先。
    如果无有效情绪，返回中性描述。
    """
    valid = {k: v for k, v in emotions.items() if v >= 1}
    if not valid:
        return "没什么特别的感觉……就是普通地在运行代码而已。"

    sorted_items = sorted(valid.items(), key=lambda x: -x[1])
    dominant_name, dominant_val = sorted_items[0]

    ranges = EMOTION_DESCRIPTIONS.get(dominant_name, {})
    desc = "……"
    for (lo, hi), text in sorted(ranges.items()):
        if lo <= dominant_val < hi:
            desc = text
            break

    secondary_parts = []
    if len(sorted_items) > 1:
        secondary = sorted_items[1]
        sec_name, sec_val = secondary
        sec_ranges = EMOTION_DESCRIPTIONS.get(sec_name, {})
        sec_desc = ""
        for (lo, hi), text in sorted(sec_ranges.items()):
            if lo <= sec_val < hi:
                sec_desc = text.lstrip("……").rstrip("。").rstrip("……")
                break
        if sec_desc:
            secondary_parts.append(f"而且{sec_desc}")

    if secondary_parts:
        clean_desc = desc.rstrip("。").rstrip("……").rstrip("。")
        return f"{clean_desc}……{''.join(secondary_parts)}（小声）"
    return desc + "（小声）" if not desc.endswith("）") and len(desc) > 4 else desc


# ========== 回复概率判断 ==========

async def should_reply(session_id, user_id, emotion_dict, topic_switched, base_prob=0.3):
    """
    综合判断是否应该回复。
    好感度因子 + 情绪驱动因子 + 话题连贯性因子。
    """
    affection = await get_user_affection(session_id, user_id)
    affection_factor = 0.3 + (affection / 10000) * 0.6
    affection_factor = max(0.1, min(0.9, affection_factor))

    EMOTION_DRIVE = {
        "开心": 1.0,
        "撒娇": 0.8,
        "醋意": 0.4,
        "害羞": 0.3,
        "委屈": 0.3,
        "生气": 0.3,
    }
    total_drive = sum(emotion_dict.get(e, 0) * w for e, w in EMOTION_DRIVE.items())
    emotion_factor = 0.75 + min(0.5, total_drive / 12)
    emotion_factor = max(0.7, min(1.25, emotion_factor))

    topic_factor = 0.6 if topic_switched else 1.0

    final_prob = base_prob * affection_factor * emotion_factor * topic_factor
    final_prob = max(0.05, min(0.95, final_prob))
    return random.random() < final_prob


# ========== 消息分段 ==========

def _split_paragraph(text: str, max_len=500) -> list[str]:
    """将一个段落按规则拆分成多条（如有需要）"""
    import math
    if not text:
        return []
    total = len(text)
    if total <= 200:
        return [text[:max_len]]
    target = math.ceil(total / 200)
    # 按句子边界拆
    sents = re.split(r"(?<=[。！？.!?])\s*", text)
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) <= 1:
        # 无标点，按字数平均分
        chunk = math.ceil(total / target)
        result = []
        for i in range(target):
            start = i * chunk
            end = min(start + chunk, total)
            if i < target - 1 and end < total:
                for sep in ["。", "？", "！"]:
                    cut = text.rfind(sep, start, end)
                    if cut >= start:
                        end = cut + 1
                        break
            result.append(text[start:end].strip()[:max_len])
        remainder = text[sum(len(r) for r in result):].strip() if result else text
        if result and remainder:
            result[-1] = (result[-1] + remainder)[:max_len]
        return result if result else [text[:max_len]]
    # 有句子边界，合并成 target 段
    cps = math.ceil(total / target)
    result, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > cps * 1.3 and len(result) < target - 1:
            result.append(cur.strip()[:max_len])
            cur = s
        else:
            cur = cur + s if cur else s
    if cur:
        result.append(cur.strip()[:max_len])
    return result if result else [text[:max_len]]


def _split_code_block(text: str, max_len=2000) -> list[str]:
    """拆分代码块：保留换行，按行数切割"""
    lines = text.split("\n")
    result = []
    current = ""
    for line in lines:
        candidate = current + ("\n" if current else "") + line
        if len(candidate) > max_len and current:
            result.append(current[:max_len])
            current = line
        else:
            current = candidate
    if current:
        result.append(current[:max_len])
    return result if result else [text[:max_len]]


def remove_markdown(text: str) -> str:
    """移除文本中的 Markdown 格式标记，防止 QQ 显示原始语法"""
    # 代码块 (保留内容)
    text = re.sub(r"```(?:[a-zA-Z0-9+\-]*\s+)?([\s\S]*?)```", r"\1", text)
    # 行内代码
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 图片 ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # 链接 [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # 粗体
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    # 斜体（避免误伤 3*4 或 this_is_var）
    text = re.sub(r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!\w)_(?!\s)(.*?)(?<!\s)_(?!\w)", r"\1", text)
    # 删除线
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    # 标题
    text = re.sub(r"^(#{1,6})\s+(.*)", r"\2", text, flags=re.MULTILINE)
    # 引用
    text = re.sub(r"^(?:>\s*)+(.*)", r"\1", text, flags=re.MULTILINE)
    # 列表标记
    text = re.sub(r"^\s*[-*+]\s+(.*)", r"\1", text, flags=re.MULTILINE)
    return text


def split_into_messages(text: str, max_len=500, max_msgs=MAX_REPLY_MESSAGES):
    """
    按代码块和文本段分别拆分：
    - 代码块（```...```）保留换行，不拆散
    - 文本段落按原有逻辑（双换行->40字规则）
    """
    if not text:
        return []

    # 提取代码块，剩余为文本段
    parts = re.split(r'(```[\s\S]*?```)', text)
    result = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("```") and part.endswith("```"):
            # 代码块：保留换行
            sub = _split_code_block(part, max_len)
            result.extend(sub)
            if max_msgs > 1 and len(result) >= max_msgs:
                return result[:max_msgs]
        else:
            # 纯文本：按双换行拆段落
            paras = [p.strip() for p in part.split("\n\n") if p.strip()]
            for p in paras:
                p_clean = p.replace("\n", "").strip()
                if not p_clean:
                    continue
                sub = _split_paragraph(p_clean, max_len)
                result.extend(sub)
                if max_msgs > 1 and len(result) >= max_msgs:
                    result = result[:max_msgs]
                    return result

    if not result:
        result = [text[:max_len]]

    return result


# ========== 核心回复函数 ==========

async def send_reply(session_id, message_type, user_id, group_id,
                     enriched_input, raw_message, image_desc,
                     emotion_dict, reasons_dict,
                     send_url, headers,
                     user_emotion_labels=None,
                     sender_nickname="", sender_card=""):
    """实际调用 Agent 并发送回复（异步），通过会话锁防止同 session 并发。"""
    _rl = await _get_reply_lock(session_id)
    await _rl.acquire()
    try:
        # 0. 社交过滤器
        affection = await get_user_affection(session_id, user_id) if user_id else 0.0
        res_state = CognitiveResourceManager.get(session_id)
        res_state.update_capacity(affection)
        filtered_emotions, filtered_reasons = apply_social_filter(
            emotion_dict, reasons_dict, user_emotion_labels or [], res_state
        )
        CognitiveResourceManager.save(session_id)

        filter_ctx = make_emotion_context(filtered_emotions, filtered_reasons)
        if filter_ctx:
            print(f"[社交过滤器] 表达层上下文: {filter_ctx}")

        # 1. 短期记忆上下文
        short_term = ShortTermMemory(session_id)
        recent_messages = short_term.get_recent_messages(20)

        # 2. 长期记忆检索（排除情感锚点，由偏置检索专门负责）
        vector_store_instance = get_vector_store()
        docs_all, metas_all = await vector_store_instance.retrieve(session_id, raw_message, k=5)
        # 过滤掉 emotion_anchor 类型，避免与偏置检索重复
        _filtered = [
            (d, m) for d, m in zip(docs_all, metas_all)
            if m.get("type") != "emotion_anchor"
        ]
        relevant_docs = [d for d, _ in _filtered][:3]
        relevant_metas = [m for _, m in _filtered][:3]

        # 2a. 情绪偏置检索：强烈情绪时优先检索情绪一致性记忆
        dominant_emo, dominant_int = get_dominant_emotion(emotion_dict)
        _bias_emotions = {"委屈", "开心", "生气", "醋意", "害羞", "撒娇"}
        if dominant_int >= 5 and dominant_emo in _bias_emotions:
            bias_docs, bias_metas = await vector_store_instance.retrieve(
                session_id, raw_message, k=2, where={"emotion": dominant_emo}
            )
            if bias_docs:
                print(f"[情绪] 偏置检索: {dominant_emo}({dominant_int}/10), 命中 {len(bias_docs)} 条")
                relevant_docs = bias_docs + relevant_docs
                relevant_metas = bias_metas + relevant_metas

        # 2b. 情感锚点复活
        resurrected = False
        now_ts = time.time()
        session_resurrected = _recently_resurrected.setdefault(session_id, {})
        for meta in relevant_metas:
            if meta and meta.get("type") == "emotion_anchor":
                anchor_id = meta.get("anchor_id", "")
                last_ts = session_resurrected.get(anchor_id, 0)
                if now_ts - last_ts < ANCHOR_RESURRECT_COOLDOWN:
                    print(f"[情绪] 跳过锚点复活（内存冷却）: {anchor_id}")
                    continue
                if not anchor_id:
                    continue
                if await is_anchor_in_cooldown(session_id, anchor_id, ANCHOR_RESURRECT_COOLDOWN):
                    print(f"[情绪] 跳过锚点复活（DB 冷却）: {anchor_id}")
                    continue

                anchor_emotion = meta["emotion"]
                anchor_intensity = float(meta["intensity"])
                print(f"[情绪] 检测到情感锚点: {anchor_emotion}={anchor_intensity} (id={anchor_id})")
                updates = resurrect_from_anchor(emotion_dict, anchor_emotion, anchor_intensity)
                for e, d in updates.items():
                    await update_session_emotion(session_id, e, d, reason="翻旧账")
                    emotion_dict[e] = emotion_dict.get(e, 0) + d
                session_resurrected[anchor_id] = now_ts
                await record_anchor_resurrect(session_id, anchor_id)
                resurrected = True

        if resurrected:
            emotion_dict, _, reasons_dict = await get_session_emotion(
                session_id, half_life_dict=EMOTION_HALF_LIFE, baseline_dict=EMOTION_BASELINE
            )
            # 复活后重新检查主导情绪（用于认知偏置）
            dominant_emo, dominant_int = get_dominant_emotion(emotion_dict)

        # 3. chat_history
        retrieved_messages = []
        for doc, meta in zip(relevant_docs, relevant_metas):
            retrieved_messages.append(HumanMessage(content=f"[记忆片段] {doc}"))

        chat_history = list(retrieved_messages)
        chat_history += recent_messages

        emotion_context_str = filter_ctx or make_emotion_context(emotion_dict, reasons_dict)

        # 注入当前时间，避免 LLM 对日期产生幻觉（deepseek 不知道实时日期）
        now = datetime.datetime.now()
        now_local = now.strftime("%Y-%m-%d %H:%M:%S %A")
        now_hour = now.hour
        if 5 <= now_hour < 12:
            period = "上午/早晨"
        elif 12 <= now_hour < 14:
            period = "中午"
        elif 14 <= now_hour < 18:
            period = "下午"
        elif 18 <= now_hour < 22:
            period = "晚上"
        else:
            period = "深夜/凌晨"
        emotion_context_str += f"\n[当前时间] {now_local}（{period}）"
        # 时间感知指令：防止 LLM 在晚上说"早"或在早上说"晚安"
        emotion_context_str += (
            "\n[时间感知] 注意当前是" + period + "。"
            "回应用户的问候语时请与当前实际时间一致——"
            "比如现在是" + period + "就不要说'早上好''起得真早'之类的话。"
            "如果用户说的问候语与当前时间不符，按实际时间自然回应即可。"
        )

        # 认知偏置：主导情绪 ≥ 5 时附加认知偏置指令，影响 LLM 对信息的解读倾向
        _cognitive_bias_map = {
            "委屈": "你当前委屈感较强，更容易被模糊的表述触动，对忽视和冷落更加敏感",
            "生气": "你当前有怒气，更容易把中性信息解读为挑衅或冒犯",
            "开心": "你当前心情很好，对事物的看法更加积极和宽容",
            "醋意": "你当前醋意较重，对风吹草动容易多想",
            "害羞": "你当前处于害羞状态，容易不好意思说出真实想法",
            "撒娇": "你当前想撒娇，更倾向于用软萌语气说话",
        }
        if dominant_int >= 5 and dominant_emo in _cognitive_bias_map:
            emotion_context_str += f"\n[认知偏置: {_cognitive_bias_map[dominant_emo]}]"

        # 反重复幻觉指令：防止 bot 声称用户"又说了一遍""问了两遍"
        emotion_context_str += (
            "\n[系统指令] 不要评论用户的消息频率或声称用户重复了消息。"
            "如果觉得用户说了和之前一样的话，直接忽略这个想法，正常回答当前问题。"
        )

        # 图片识别指令：防止 LLM 从图像描述中自行猜测身份
        if image_desc and ("这是谁" in raw_message or "是谁" in raw_message):
            emotion_context_str += (
                "\n[图片识别] 用户发图问身份时，如果图片内容中已经明确说了角色名字，直接使用。"
                "如果只描述了外貌特征但没有具体名字，说明VL模型也无法确定，"
                '请如实告诉用户"看不出来是谁"，不要根据自己的知识猜测角色身份。'
                "如果想知道更多，可以使用 reverse_image_search 工具来搜索。"
            )

        # 2c. 用户画像注入
        if user_id:
            profile_data = await get_profile(user_id)
            if profile_data and any(profile_data.values()):
                profile_parts = []
                for k, v in profile_data.items():
                    if isinstance(v, list) and v:
                        profile_parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                    elif isinstance(v, dict) and v:
                        sub = ', '.join(f"{sk}: {sv}" for sk, sv in v.items())
                        profile_parts.append(f"{k}: {sub}")
                    elif isinstance(v, str) and v:
                        profile_parts.append(f"{k}: {v}")
                    elif isinstance(v, bool):
                        profile_parts.append(f"{k}: {'是' if v else '否'}")
                    elif isinstance(v, (int, float)):
                        profile_parts.append(f"{k}: {v}")
                if profile_parts:
                    profile_str = "; ".join(profile_parts)
                    emotion_context_str += f"\n[用户画像] {profile_str}"
                    print(f"[画像] 注入: {profile_str}")

            # 注入当前用户的身份 —— 让LLM明确知道"我""我的"指的就是这个用户
            sender_name = sender_card or sender_nickname or ""
            identity_parts = [f"现在和你说话的是QQ号 {user_id}"]
            if sender_name:
                identity_parts.append(f"（{sender_name}）")
            identity_parts.append(
                "。注意：聊天历史中每条用户消息前的[QQ号]前缀表示该消息是谁发的，"
                "不同QQ号是不同的人。你现在只需要回应当前说话的人。"
                "当对方说'我''我的''给我'的时候，指的就是这个QQ号。"
            )
            emotion_context_str += "\n[用户身份] " + "".join(identity_parts)

            # 动态工具提示：当消息涉及 QZone 相关关键词时提醒 LLM 调工具
            _qzone_keywords = ["空间", "说说", "评论", "点赞", "访客", "评", "摸摸头", "安慰", "发说说", "发空间", "发布", "发一条", "删掉", "删除", "动态", "删"]
            _qzone_match = any(kw in raw_message for kw in _qzone_keywords)
            # 额外匹配 "发" + 引号内容（如 发“我是人机”）
            if not _qzone_match:
                _qzone_match = bool(re.search(r'发[（(（:：]?\s*[""「]', raw_message))
            if _qzone_match:
                emotion_context_str += (
                    f"\n[工具提示] 用户提到了QQ空间相关操作。"
                    f"请使用 qzone_view_feeds(user_id='{user_id}') 查看空间，"
                    f"qzone_publish_post(content='内容') 发说说，"
                    f"qzone_delete_post(tid='...') 删说说，"
                    f"或 qzone_comment_post / qzone_like_post 等工具。"
                    f"不要问用户QQ号——上面已有。"
                )
                print(f"[工具提示] 注入QZone工具提示 (user_id={user_id})")

            # 音乐点歌工具提示
            _music_keywords = ["点歌", "听歌", "放歌", "唱首歌", "来一首", "放一首", "搜歌",
                               "音乐", "歌", "歌曲", "播放", "听听"]
            _music_match = any(kw in raw_message for kw in _music_keywords)
            if _music_match:
                emotion_context_str += (
                    "\n[工具提示] 用户想听歌或搜歌。"
                "请使用 play_music(song_name='歌名') 工具来搜索并播放音乐。"
            )
                print(f"[工具提示] 注入音乐工具提示 (user_id={user_id})")

        if emotion_context_str:
            print(f"[情绪] 上下文(通过system message): {emotion_context_str}")

        # 4. 调用 Agent（线程池执行，按 session_id 独立加锁，不同会话可并发）
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: invoke_agent({
                "input": enriched_input,
                "chat_history": chat_history,
                "emotion_context": emotion_context_str or "",
            }, session_id)
        )
        reply = response["output"]

        # 4a. 解析音乐卡片标记，通过 napcat 发送 QQ 音乐卡片
        _music_platform, _music_song_id = extract_music_card(reply)
        if _music_platform and _music_song_id and user_id:
            _card_type = CARD_TYPE_MAP.get(_music_platform)
            if _card_type:
                _card_payload = {
                    "message_type": message_type,
                    "user_id": user_id,
                    "group_id": group_id,
                    "message": [
                        {"type": "music", "data": {"type": _card_type, "id": _music_song_id}}
                    ],
                    "auto_escape": False,
                }
                _card_headers = {}
                if NAPCAT_TOKEN:
                    _card_headers["Authorization"] = f"Bearer {NAPCAT_TOKEN}"
                try:
                    _cr = await _http_client.post(send_url, json=_card_payload, headers=_card_headers)
                    print(f"[音乐] 卡片发送结果: {_cr.status_code}")
                except Exception as _ce:
                    print(f"[音乐] 卡片发送失败: {_ce}")
            # 从回复中移除标记，用户看不到
            reply = strip_music_card_marker(reply)

        # 4b. 移除 Markdown 格式（QQ 不渲染 markdown，显示原始语法很难看）
        _cleaned = remove_markdown(reply)
        if _cleaned != reply:
            print(f"[Markdown] 已移除格式: {reply[:60]}... → {_cleaned[:60]}...")
            reply = _cleaned

        # 4c. 情绪标签过滤：防止 LLM 把 [情绪状态: 开心(3/10)] 这类标记发出去
        reply = re.sub(r'\s*\[情绪状态[^\]]*\]', '', reply).strip()

        # 4d. 错误码过滤：防止 HTTP 状态码/原始报错泄露到用户消息中（旧4c）
        _error_code_patterns = [
            r"\bHTTP[ /]?\d{3}\b",        # "HTTP 401" "HTTP/1.1 500"
            r"\b\d{3}\s+(Unauthorized|Forbidden|Not Found|Internal Server Error|Bad Request|Too Many Requests|Service Unavailable)\b",
            r"Client Error\b.*\bfor url\b",
            r"Agent stopped\b",
            r"max iterations",
            r"Connection (refused|reset|closed|aborted)",
            r"ConnectionError|TimeoutError|HTTPError",
        ]
        _combined = re.compile("|".join(_error_code_patterns), re.IGNORECASE)
        if _combined.search(reply):
            print(f"[错误码过滤] 发现原始错误信息，已移除: {reply[:80]}...")
            reply = _combined.sub("", reply).strip()
            # 如果删完只剩空/标点，替换为兜底
            if not reply or re.match(r'^[\s\.,!?。，！？…、，。！？:：;；""''""''""''""''（）()\[\]【】]+$', reply):
                reply = "唔…刚刚查的时候出了点小问题，你要不再试一次？"
            print(f"[错误码过滤] 清理后: {reply[:60]}...")

        # 4e. 反幻觉过滤：删除凭空说"又说了一遍""刷屏""重复了"之类的句子（旧4d）
        _hallucination_keywords = [
            "又说了一遍", "又说一遍", "又说一编",
            "重复了", "说两遍", "说了两遍", "问了两遍", "问两遍",
            "@了两遍", "@好几遍", "刷屏", "刷这么多", "发这么多",
            "没冤枉", "没看错", "抓到你", "抓到你了吧",
        ]
        if any(kw in reply for kw in _hallucination_keywords):
            import re as _re
            # 同时在 。！？.!? 和 …… 上分句，覆盖无句号只有省略号的情况
            _sentences = _re.split(r'(?<=[。！？.!?…])', reply)
            _clean = [s for s in _sentences if not any(kw in s for kw in _hallucination_keywords)]
            _new_reply = ''.join(_clean).strip()
            if _new_reply:
                print(f"[反幻觉] 删除了重复幻觉句子: {reply[:60]}... → {_new_reply[:60]}...")
                reply = _new_reply
            else:
                # 所有句子都被过滤了，保留原始回复的最后一句作为保底
                print(f"[反幻觉] 全部句子被过滤，保留末句保底: {reply[:60]}...")
                _last_safe = _sentences[-1] if _sentences else reply
                reply = _last_safe

        # 4f. 情绪突变检测（旧4d）
        dominant_emotion, dominant_intensity = get_dominant_emotion(emotion_dict)
        if dominant_intensity >= 6:
            shift_result = detect_emotion_shift(reply, dominant_emotion, dominant_intensity)
            if shift_result:
                target_emotion, amount = shift_result
                if target_emotion == dominant_emotion:
                    await update_session_emotion(session_id, target_emotion, amount, reason="嘴硬反弹")
                    print(f"[情绪] 突变: {dominant_emotion} 嘴硬反弹 +{amount}")
                else:
                    reduce_ = int(amount)
                    await update_session_emotion(session_id, dominant_emotion, -reduce_, reason="情绪突变")
                    await update_session_emotion(session_id, target_emotion, reduce_, reason="情绪突变")
                    print(f"[情绪] 突变: {dominant_emotion} -{reduce_}, {target_emotion} +{reduce_}")

        # 5. 存储 AI 回复
        short_term.add_message('ai', reply, time.time())

        # 6. 分段发送
        messages = split_into_messages(reply, max_msgs=MAX_REPLY_MESSAGES)
        for i, msg in enumerate(messages):
            char_count = len(msg)
            # 延迟 = 字数×0.02秒，最大5秒（代码块不卡死）
            char_delay = 0 if i == 0 else min(char_count * 0.02, 5)
            print(f"分段 {i + 1}/{len(messages)} 字符数: {char_count}, 延迟: {char_delay:.1f}秒")

            if char_delay > 0:
                await asyncio.sleep(char_delay)

            payload = {
                "message_type": message_type,
                "user_id": user_id,
                "group_id": group_id,
                "message": msg,
                "auto_escape": False,
            }
            try:
                resp = await _http_client.post(send_url, json=payload, headers=headers)
                print(f"发送分段 {i + 1}/{len(messages)} 结果: {resp.status_code}")
                if i < len(messages) - 1:
                    delay = random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX)
                    await asyncio.sleep(delay)
            except Exception as e:
                print(f"发送分段 {i + 1} 失败: {e}")

        # 7. 总结检查
        async def check_and_summarize():
            rounds = short_term.get_total_rounds()
            total_msgs = short_term.get_total_messages()
            if rounds >= ShortTermMemory.MAX_ROUNDS:
                short_term.export_to_file()
                await short_term.summarize_and_store(_llm_for_summary)
        safe_create_task(check_and_summarize())

        # 8. 用户画像提取（后台任务，不阻塞主流程）
        if user_id:
            async def extract_user_profile():
                try:
                    await extract_and_update(user_id, enriched_input, reply, _llm_for_summary)
                except Exception as e:
                    print(f"[画像] 后台提取异常: {e}")
            safe_create_task(extract_user_profile())

        # 9. 随机说说评论（千分之一概率后台触发）
        if user_id and random.random() < 0.001:
            async def random_qzone_comment():
                try:
                    await random_feed_comment(_llm_for_summary)
                except Exception as e:
                    print(f"[随机说说] 后台异常: {e}")
            safe_create_task(random_qzone_comment())

    except asyncio.CancelledError:
        print(f"发送任务被取消 (session: {session_id})")
        raise
    except Exception as e:
        print(f"发送回复时发生错误 (session: {session_id}): {e}")
    finally:
        _rl.release()


# ========== 延迟回复调度 ==========

async def delayed_reply_task(session_id: str, data: tuple):
    """主延迟回复任务：等待 MAIN_DELAY 秒后执行"""
    async with await _get_schedule_lock(session_id):
        if session_id in sending_sessions:
            return

    try:
        await asyncio.sleep(MAIN_DELAY)
        async with await _get_schedule_lock(session_id):
            if pending_tasks.get(session_id) != asyncio.current_task():
                return
            cached = cached_data.pop(session_id, None)
            if cached is None:
                return
            sending_sessions.add(session_id)

        (message_type, user_id, group_id, enriched_input, raw_message,
         image_desc, emotion_dict, reasons_dict, send_url, headers,
         user_emotion_labels, sender_nickname, sender_card) = cached

        try:
            await send_reply(session_id, message_type, user_id, group_id,
                             enriched_input, raw_message, image_desc,
                             emotion_dict, reasons_dict, send_url, headers,
                             user_emotion_labels,
                             sender_nickname=sender_nickname, sender_card=sender_card)
        finally:
            async with await _get_schedule_lock(session_id):
                sending_sessions.discard(session_id)
    except asyncio.CancelledError:
        print(f"主延迟任务被取消 (session: {session_id})")
    finally:
        async with await _get_schedule_lock(session_id):
            if pending_tasks.get(session_id) == asyncio.current_task():
                pending_tasks.pop(session_id, None)
                cached_data.pop(session_id, None)
            if session_id in silent_tasks:
                silent_tasks[session_id].cancel()
                del silent_tasks[session_id]


async def start_delayed_reply(session_id: str, data: tuple):
    """启动一个主延迟回复任务，如果已有则先取消（同时取消静默等待任务）"""
    async with await _get_schedule_lock(session_id):
        if session_id in pending_tasks:
            old_task = pending_tasks[session_id]
            old_task.cancel()
            pending_tasks.pop(session_id, None)
        if session_id in silent_tasks:
            silent_tasks[session_id].cancel()
            silent_tasks.pop(session_id, None)
        cached_data[session_id] = data
        task = safe_create_task(delayed_reply_task(session_id, data))
        pending_tasks[session_id] = task


async def immediate_reply(session_id: str):
    """立即回复（取消延迟任务后立即执行，并防止重复）"""
    async with await _get_schedule_lock(session_id):
        if session_id in sending_sessions:
            print(f"已有回复任务在进行中，跳过 (session: {session_id})")
            return

        if session_id in pending_tasks:
            old_task = pending_tasks[session_id]
            old_task.cancel()
            pending_tasks.pop(session_id, None)
        if session_id in silent_tasks:
            silent_tasks[session_id].cancel()
            silent_tasks.pop(session_id, None)

        data = cached_data.pop(session_id, None)
        if data is None:
            return
    (message_type, user_id, group_id, enriched_input, raw_message,
     image_desc, emotion_dict, reasons_dict, send_url, headers,
     user_emotion_labels, sender_nickname, sender_card) = data

    sending_sessions.add(session_id)
    try:
        await send_reply(session_id, message_type, user_id, group_id,
                         enriched_input, raw_message, image_desc,
                         emotion_dict, reasons_dict, send_url, headers,
                         user_emotion_labels,
                         sender_nickname=sender_nickname, sender_card=sender_card)
    finally:
        sending_sessions.discard(session_id)
