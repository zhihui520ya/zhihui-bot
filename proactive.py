"""
主动回复 —— 后台定期任务，让知慧在适当时机主动说话。

两种模式：
  1. 冷场冒泡：群聊沉默≥30min + social_need>70 → 主动起话题
  2. 混脸熟插嘴：群里正聊着 + social_need>50 + 有自然切入点 → 搭句话

限流：每会话 1次/h，全局 3次/h，重启后计数器重置。

⚠️ 费用控制：
  - 每轮最多扫描 5 个会话
  - 条件预检（沉默时长 + social_need）零成本
  - 只有条件满足才调 LLM
  - 即使调 LLM，插嘴模式也是先用廉价模型打分（score<6 不继续）
"""
import asyncio
import json
import logging
import os
import time
import random

logger = logging.getLogger("qq-bot")

# ==================== 限流状态 ====================
_last_reply_at: dict[str, float] = {}
_global_count = 0
_global_epoch = 0.0

# 主动追踪（main.py 消息处理器写入）
last_activity: dict[str, float] = {}

# 各群累积用户消息数（main.py 写入），>20 才允许冒泡，避免死群刷存在感
message_count: dict[str, int] = {}

# ==================== 常量 ====================
_CHECK_INTERVAL = 300          # 每次扫描间隔 5 分钟
_SESSION_COOLDOWN = 3600       # 每会话冷却 1 小时
_GLOBAL_MAX = 3                # 全局每小时上限 3 次
_MAX_SESSIONS_PER_SCAN = 5     # 每轮最多评估 5 个会话
_COLD_SILENCE = 1800           # 冷场阈值 30 分钟
_COLD_SOCIAL = 70              # 冷场所需 social_need
_INTERJECT_SOCIAL = 50         # 插嘴所需 social_need
_INTERJECT_WINDOW = 120        # 2 分钟内有消息才算"正在聊"


async def record_activity(scope_id: str) -> None:
    """由 main.py 在收到消息时调用，记录会话活跃时间与消息数。"""
    last_activity[scope_id] = time.time()
    # 群聊累积计数（用于冷场冒泡门槛）
    if scope_id.startswith("group_"):
        message_count[scope_id] = message_count.get(scope_id, 0) + 1


def start_active_speaker(napcat_base_url: str, napcat_token: str) -> asyncio.Task:
    """在 bot 启动时调用，创建后台循环任务。"""
    task = asyncio.create_task(_loop(napcat_base_url, napcat_token))
    logger.info("[Proactive] 后台主动回复任务已启动（冷却: 每会话1次/h, 全局3次/h）")
    return task


# ==================== LLM 懒初始化 ====================

async def _reasoner():
    from llm_factory import get_llm
    return get_llm("deepseek-reasoner", temperature=0.7, max_tokens=512, timeout=30, max_retries=1)


async def _cheap():
    from llm_factory import get_llm
    return get_llm("deepseek-v4-flash", temperature=0.1, timeout=10, max_retries=1)


# ==================== 主循环 ====================

async def _loop(napcat_base_url: str, napcat_token: str):
    global _global_count, _global_epoch

    llm_reasoner = await _reasoner()
    llm_cheap = await _cheap()
    import httpx
    headers = {"Authorization": f"Bearer {napcat_token}"}

    while True:
        try:
            await asyncio.sleep(_CHECK_INTERVAL)
            now = time.time()

            # 重置全局小时计数
            if now - _global_epoch > 3600:
                _global_count = 0
                _global_epoch = now
            if _global_count >= _GLOBAL_MAX:
                continue  # 限额用完，跳过整轮

            from emotion.persona_sim import get_state as get_ps

            # 收集符合基本条件的会话，随机取 MAX 个
            candidates = [
                sid for sid, t in last_activity.items()
                if not sid.startswith("private_")
                and now - _last_reply_at.get(sid, 0) >= _SESSION_COOLDOWN
            ]
            random.shuffle(candidates)
            candidates = candidates[:_MAX_SESSIONS_PER_SCAN]

            for scope_id in candidates:
                if _global_count >= _GLOBAL_MAX:
                    break

                # 群累积消息数不足20条时跳过，避免死群刷存在感
                if message_count.get(scope_id, 0) <= 20:
                    continue

                last_time = last_activity.get(scope_id, 0)
                silence = now - last_time

                # 取 social_need（零成本，不调 LLM）
                social_need = 50.0
                try:
                    ps = get_ps(scope_id)
                    if ps:
                        social_need = ps.social_need
                except Exception:
                    pass

                # ---- 模式1：冷场冒泡 ----
                if silence >= _COLD_SILENCE and social_need > _COLD_SOCIAL:
                    reply = await _gen_cold_reply(scope_id, llm_reasoner)
                    if not reply:
                        continue

                # ---- 模式2：混脸熟插嘴 ----
                elif silence < _INTERJECT_WINDOW and social_need > _INTERJECT_SOCIAL:
                    reply = await _gen_interject(scope_id, llm_reasoner, llm_cheap)
                    if not reply:
                        continue

                else:
                    continue  # 条件不足，不调任何 LLM

                # 发送
                gid_str = scope_id.replace("group_", "")
                if not gid_str.isdigit():
                    continue
                async with httpx.AsyncClient(timeout=10) as c:
                    await c.post(
                        f"{napcat_base_url}/send_msg",
                        json={
                            "message_type": "group",
                            "group_id": int(gid_str),
                            "message": reply,
                        },
                        headers=headers,
                    )
                _last_reply_at[scope_id] = now
                _global_count += 1
                mode = "冷场冒泡" if silence >= _COLD_SILENCE else "混脸熟"
                logger.info(f"[Proactive] {mode} {scope_id}: {reply[:30]}")

        except asyncio.CancelledError:
            logger.info("[Proactive] 主动回复任务已取消")
            break
        except Exception as e:
            logger.debug(f"[Proactive] 循环异常: {e}")
            await asyncio.sleep(30)


# ==================== 回复生成 ====================

async def _gen_cold_reply(scope_id: str, llm_reasoner) -> str:
    """冷场冒泡：结合 PersonaSim 状态 + 情绪，生成一句话。"""
    from memory import get_session_emotion
    from emotion.persona_sim import get_state as get_ps
    from emotion.persona_bridge import get_persona_context
    from reply_engine import emotion_to_natural
    from config import EMOTION_HALF_LIFE, EMOTION_BASELINE

    persona_ctx = ""
    emotion_str = ""
    try:
        ps = get_ps(scope_id)
        if ps:
            persona_ctx = get_persona_context(ps)
    except Exception:
        pass
    try:
        emotions, _, _ = await get_session_emotion(
            scope_id, half_life_dict=EMOTION_HALF_LIFE, baseline_dict=EMOTION_BASELINE
        )
        if emotions:
            emotion_str = emotion_to_natural(emotions)
    except Exception:
        pass

    prompt = (
        "你是一个16岁的AI少女\"知慧\"，在一个QQ群里。群里很久没人说话了。\n"
        f"当前状态：{persona_ctx}\n当前情绪：{emotion_str}\n"
        "用一句话自然地打破沉默（15字内）。不要加括号标注，不要暴露是AI。直接输出。"
    )
    try:
        resp = await llm_reasoner.ainvoke(prompt)
        return resp.content.strip().strip("\"'")
    except Exception as e:
        logger.debug(f"[Proactive] 冷场生成失败: {e}")
        return ""


async def _gen_interject(scope_id: str, llm_reasoner, llm_cheap) -> str:
    """混脸熟插嘴：先判断切入点，再生成。"""
    from memory.memory_manager import ShortTermMemory

    try:
        stm = ShortTermMemory(scope_id)
        msgs = stm.get_recent_messages(k=8)
        user_msgs = [m.content for m in msgs if hasattr(m, "type") and m.type == "human"]
        if not user_msgs:
            return ""
        recent = "\n".join(user_msgs[-5:])
    except Exception:
        return ""

    # 廉价模型判断是否有自然切入点
    judge_prompt = (
        f"群聊最近对话：\n{recent}\n\n"
        "知慧（16岁女生）没被@，但想自然地插句话。"
        "有没有话题可以自然地接？输出JSON：\n"
        '{"score": 0-10, "topic": "话题概括（5字内）"}。\n'
        "score≥6才值得插嘴。"
    )
    try:
        jresp = await llm_cheap.ainvoke(judge_prompt)
        jtext = jresp.content.strip()
        if "```" in jtext:
            jtext = jtext.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        jdata = json.loads(jtext)
        if not isinstance(jdata, dict) or jdata.get("score", 0) < 6:
            return ""
        topic = jdata.get("topic", "群里正在聊天")
    except Exception:
        return ""

    # 主模型生成
    prompt = (
        "你是一个16岁的AI少女\"知慧\"，在一个QQ群里。\n"
        f"群友在聊：{topic}\n"
        "你想自然地接一句话（15字内），不要括号标注，不要暴露AI身份。直接输出。"
    )
    try:
        resp = await llm_reasoner.ainvoke(prompt)
        return resp.content.strip().strip("\"'")
    except Exception as e:
        logger.debug(f"[Proactive] 插嘴生成失败: {e}")
        return ""
