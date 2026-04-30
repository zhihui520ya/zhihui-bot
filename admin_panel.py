"""
管理员调试面板 —— 处理所有 /admin 指令。
"""
import time
import os
import json
import sqlite3
import datetime as dt_module
import asyncio
import requests
from config import (
    EMOTION_HALF_LIFE, EMOTION_BASELINE,
    COGNITIVE_RECOVERY_RATE, NAPCAT_TOKEN, NAPCAT_BASE_URL, DATA_DIR,
    ADMIN_QQ,
)
from memory.emotion_store import (
    get_session_emotion, set_session_emotion, clear_session_emotion,
    get_user_affection, get_anchored_labels, record_anchor_label,
    _get_conn,
)
from memory.user_profile import delete_profile, get_profile
from memory.memory_manager import ShortTermMemory
from emotion.social_filter import CognitiveResourceManager
from api import reset_session

# 管理员 QQ 号列表（从 config.py 读取，通过 .env 的 ADMIN_QQ 设置）
# 重置会话确认缓存（session_id -> {"target": str, "time": float, "desc": str}）
pending_confirms: dict[str, dict] = {}

# 全局 vector_store 引用（延迟注入，避免循环导入）
_vector_store = None


def _set_vector_store(vs):
    global _vector_store
    _vector_store = vs


# ================================================================

async def _reset_session_data(target_session_id: str) -> str:
    """重置指定会话的所有数据（统一 api.reset_session 入口），返回结果描述"""
    try:
        cleaned = await reset_session(target_session_id)
        return f"会话 {target_session_id} 已重置: {', '.join(cleaned)}"
    except Exception as e:
        return f"重置 {target_session_id} 失败: {e}"


async def _resolve_member_name(group_id: int, name: str) -> int | None:
    """根据群昵称或昵称在群成员中查找QQ号"""
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: requests.get(
                f"{NAPCAT_BASE_URL}/get_group_member_list",
                params={"group_id": group_id},
                headers={"Authorization": f"Bearer {NAPCAT_TOKEN}"},
                timeout=5,
            )
        )
        members = resp.json().get("data", [])
        name_lower = name.lower().replace(" ", "")
        for m in members:
            card = (m.get("card") or "").lower().replace(" ", "")
            nickname = (m.get("nickname") or "").lower().replace(" ", "")
            if name_lower in card or name_lower in nickname:
                return int(m["user_id"])
        return None
    except Exception as e:
        print(f"[管理员] 解析群成员名称失败: {e}")
        return None


async def _send_admin_reply(reply: str, message_type: str, user_id: int, group_id: int,
                            send_url: str, headers: dict) -> bool:
    """发送管理员回复并返回 True"""
    payload = {
        "message_type": message_type,
        "user_id": user_id,
        "group_id": group_id,
        "message": reply,
        "auto_escape": False,
    }
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: requests.post(send_url, json=payload, headers=headers, timeout=5)
        )
        print(f"[管理员] 回复发送: {resp.status_code}")
    except Exception as e:
        print(f"[管理员] 发送失败: {e}")
    return True


async def handle_admin_command(session_id: str, user_id: int, message_type: str, group_id: int,
                               raw_message: str, send_url: str, headers: dict,
                               llm_for_emotion=None) -> bool:
    """
    处理管理员调试指令。返回 True 表示已处理，调用方应跳过后续流程。
    指令格式: /admin <子命令> [参数...]
    """
    text = raw_message.strip()
    if not text.startswith("/admin"):
        return False
    if user_id not in ADMIN_QQ:
        print(f"[管理员] 非管理员用户 {user_id} 尝试使用 /admin，已忽略")
        return True

    # 自动重建 DB
    os.makedirs(f"{DATA_DIR}/memories", exist_ok=True)
    try:
        with sqlite3.connect(f"{DATA_DIR}/memories/memory.db") as tmp:
            tmp.execute("SELECT 1 FROM session_emotions LIMIT 1")
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        print("[管理员] 检测到 memory.db 损坏或缺失，自动重建")
        _get_conn()

    parts = text.split()
    sub = parts[1] if len(parts) > 1 else "help"

    # ==================== 资源系统 ====================
    if sub in ("资源", "res", "状态", "status"):
        sub2 = parts[2] if len(parts) > 2 else ""

        if sub2 in ("", "状态", "status"):
            emotion_dict, _, reasons_dict = await get_session_emotion(
                session_id, half_life_dict=EMOTION_HALF_LIFE, baseline_dict=EMOTION_BASELINE
            )
            affection = await get_user_affection(session_id, user_id)
            state = CognitiveResourceManager.get(session_id)
            state.update_capacity(affection)
            filtered, _ = _apply_social_filter_local(emotion_dict, reasons_dict, [], state)
            lines = [
                "==== 管理员状态 ====",
                f"情绪(原始): { {k: round(v, 2) for k, v in emotion_dict.items()} }",
                f"情绪(过滤): { {k: round(v, 2) for k, v in filtered.items()} if filtered else '(空)' }",
                f"好感度: {affection:.0f}",
                f"认知资源: {state.current:.1f}/{state.max_capacity:.0f} ({state.ratio:.0%})",
                f"爆发模式: {'是' if state.burst_mode else '否'}",
                f"抑制计数: {state.suppression_count}",
            ]
            reply = "\n".join(lines)

        elif sub2 in ("set", "设置"):
            if len(parts) >= 3:
                try:
                    val = float(parts[3])
                    state = CognitiveResourceManager.get(session_id)
                    state.update_capacity(await get_user_affection(session_id, user_id))
                    val = max(0, min(state.max_capacity, val))
                    state.current = val
                    CognitiveResourceManager.save(session_id)
                    reply = f"认知资源已设为 {val:.1f}/{state.max_capacity:.0f}"
                except ValueError:
                    reply = f"参数错误: '{parts[3]}' 不是有效数字"
            else:
                reply = "用法: /admin 资源 set <值>"

        elif sub2 in ("rate", "速率"):
            if len(parts) >= 3:
                try:
                    val = float(parts[3])
                    reply = (
                        f"注意: 恢复速率默认从 config.py 读取，重启后重置。\n"
                        f"当前运行时值: {COGNITIVE_RECOVERY_RATE:.6f} → 尝试设置 {val:.6f}\n"
                        f"（实际持久化需修改 config.py / .env 中的 COGNITIVE_RECOVERY_RATE）"
                    )
                except ValueError:
                    reply = f"参数错误: '{parts[3]}' 不是有效数字"
            else:
                reply = f"当前恢复速率: {COGNITIVE_RECOVERY_RATE:.6f}/秒"

        elif sub2 in ("重置", "reset"):
            state = CognitiveResourceManager.get(session_id)
            state.current = state.max_capacity
            state.burst_mode = False
            state.suppression_count = 0
            CognitiveResourceManager.save(session_id)
            reply = f"认知资源已重置至满 ({state.current:.1f}/{state.max_capacity:.0f})"

        elif sub2 in ("爆发", "burst"):
            state = CognitiveResourceManager.get(session_id)
            state.burst_mode = not state.burst_mode
            state.current = max(0.0, state.current - 10)
            CognitiveResourceManager.save(session_id)
            reply = f"爆发模式已切换为: {'开启' if state.burst_mode else '关闭'} (资源: {state.current:.1f}/{state.max_capacity:.0f})"

        elif sub2 in ("exitburst", "退爆"):
            state = CognitiveResourceManager.get(session_id)
            state.burst_mode = False
            CognitiveResourceManager.save(session_id)
            reply = f"爆发模式已强制退出 (资源: {state.current:.1f}/{state.max_capacity:.0f})"

        else:
            reply = (f"未知资源子命令: {sub2}\n"
                     f"可用: 状态 / set <值> / rate <值> / 重置 / 爆发 / exitburst")

    # ==================== 好感度 ====================
    elif sub in ("好感", "aff"):
        if len(parts) >= 3:
            target_str = parts[2]

            # 判断是纯数字 → 可能是好感度值或QQ号
            if target_str.isdigit() or (target_str.startswith("-") and target_str[1:].isdigit()):
                num = int(target_str)
                if num > 10000:
                    # 大于好感度上限 → 当作 QQ 号查询
                    affection = await get_user_affection(session_id, num)
                    reply = f"用户 {num} 的当前好感度: {affection:.0f}"
                else:
                    # ≤10000 → 设置好感度
                    val = max(0, min(10000, float(target_str)))
                    now_str = dt_module.datetime.utcnow().isoformat()
                    conn = sqlite3.connect(f"{DATA_DIR}/memories/memory.db")
                    conn.execute(
                        "INSERT OR REPLACE INTO user_affection (session_id, user_id, affection, last_updated) VALUES (?, ?, ?, ?)",
                        (session_id, user_id, round(val, 2), now_str),
                    )
                    conn.commit()
                    conn.close()
                    state = CognitiveResourceManager.get(session_id)
                    state.update_capacity(val)
                    CognitiveResourceManager.save(session_id)
                    reply = f"好感度已设为 {val:.0f}，资源容量: {state.max_capacity:.0f}"
            else:
                # 非纯数字 → 按昵称查询（仅群聊）
                if message_type != "group":
                    reply = "私聊查询他人请使用QQ号: /admin 好感 <QQ号>"
                    return await _send_admin_reply(reply, message_type, user_id, group_id, send_url, headers)
                target_qq = await _resolve_member_name(group_id, target_str)
                if not target_qq:
                    reply = f"未在群内找到匹配的用户: {target_str}"
                    return await _send_admin_reply(reply, message_type, user_id, group_id, send_url, headers)
                affection = await get_user_affection(session_id, target_qq)
                reply = f"用户 {target_str}({target_qq}) 的当前好感度: {affection:.0f}"
        else:
            affection = await get_user_affection(session_id, user_id)
            reply = f"当前好感度: {affection:.0f}"

    # ==================== 情绪 ====================
    elif sub in ("情绪", "emo"):
        if len(parts) >= 4:
            emo_name = parts[2]
            try:
                emo_val = float(parts[3])
                emo_val = max(0, min(10, emo_val))
            except ValueError:
                reply = f"参数错误: '{parts[3]}' 不是有效数字"
                return await _send_admin_reply(reply, message_type, user_id, group_id, send_url, headers)
            valid_emotions = {"开心", "生气", "委屈", "害羞", "醋意", "撒娇"}
            if emo_name not in valid_emotions:
                reply = f"无效情绪标签。可选: {', '.join(valid_emotions)}"
                return await _send_admin_reply(reply, message_type, user_id, group_id, send_url, headers)
            emotion_dict, _, reasons_dict = await get_session_emotion(session_id)
            if emo_val > 0:
                emotion_dict[emo_name] = emo_val
            else:
                emotion_dict.pop(emo_name, None)
            await set_session_emotion(session_id, emotion_dict, reasons_dict)
            reply = f"情绪 {emo_name} 已设为 {emo_val:.1f}"
        else:
            emotion_dict, _, reasons_dict = await get_session_emotion(
                session_id, half_life_dict=EMOTION_HALF_LIFE, baseline_dict=EMOTION_BASELINE
            )
            reply = f"当前情绪: { {k: round(v, 2) for k, v in emotion_dict.items()} }"

    elif sub in ("清情绪", "clearemo"):
        await clear_session_emotion(session_id)
        reply = "情绪已清空"

    # ==================== 检索 ====================
    elif sub in ("检索", "search"):
        keyword = " ".join(parts[2:]) if len(parts) > 2 else ""
        if not keyword:
            reply = "用法: /admin 检索 <关键词>"
        else:
            try:
                docs, metas = await _vector_store.retrieve(session_id, keyword, k=5)
                if not docs:
                    reply = f"未检索到与「{keyword}」相关的长期记忆"
                else:
                    lines = [f"检索「{keyword}」结果 ({len(docs)} 条):"]
                    for i, (doc, meta) in enumerate(zip(docs, metas)):
                        lines.append(f"{i + 1}. {doc[:150]}...")
                    reply = "\n".join(lines)
            except Exception as e:
                reply = f"检索失败: {e}"

    # ==================== 锚点 ====================
    elif sub in ("锚点", "anchors"):
        try:
            labels = await get_anchored_labels(session_id)
            if not labels:
                reply = "当前会话没有情感锚点记录"
            else:
                reply = f"情感锚点标签: {', '.join(sorted(labels))}"
        except Exception as e:
            reply = f"获取锚点失败: {e}"

    # ==================== 清空短期记忆 ====================
    elif sub in ("清记忆", "clearmem"):
        ShortTermMemory(session_id).clear()
        reply = "短期记忆已清空"

    # ==================== 查看上下文 ====================
    elif sub in ("上下文", "context"):
        try:
            limit = int(parts[2]) if len(parts) > 2 else 10
            limit = max(1, min(50, limit))
            short_term = ShortTermMemory(session_id)
            rows = short_term._execute(
                "SELECT message, created_at FROM message_store ORDER BY id DESC LIMIT ?",
                (limit,), fetchall=True
            )
            if not rows:
                reply = "短期记忆为空"
            else:
                lines = [f"最近 {len(rows)} 条消息:"]
                for row in reversed(rows):
                    msg_data = json.loads(row['message'])
                    role = "用户" if msg_data['type'] == 'human' else "知慧"
                    content = msg_data['data']['content'][:80]
                    lines.append(f"[{role}] {content}")
                reply = "\n".join(lines)
        except Exception as e:
            reply = f"获取上下文失败: {e}"

    # ==================== 调试情绪分析 ====================
    elif sub in ("prompt", "分析"):
        text_to_analyze = " ".join(parts[2:]) if len(parts) > 2 else ""
        if not text_to_analyze:
            reply = "用法: /admin prompt <文本>"
        else:
            try:
                from emotion.emotion_analyzer import analyze_user_emotion, check_trigger_words
                trigger = check_trigger_words(text_to_analyze)
                result_line = f"关键词触发: {trigger}" if trigger else "关键词触发: 无命中"
                emo_list = await analyze_user_emotion(text_to_analyze, [], llm_for_emotion)
                if emo_list:
                    emo_str = " ".join(f"{e}({i})[{c:.1f}]" for e, i, c in emo_list)
                else:
                    emo_str = "无"
                reply = (
                    f"分析文本: {text_to_analyze}\n"
                    f"{result_line}\n"
                    f"LLM分析: {emo_str}"
                )
            except Exception as e:
                reply = f"分析失败: {e}"

    # ==================== 重置会话 ====================
    elif sub in ("重置会话", "resetsession"):
        sub2 = parts[2] if len(parts) > 2 else ""

        if sub2 == "private":
            target = f"private_{user_id}"
            reply = await _reset_session_data(target)

        elif sub2 == "group":
            if message_type == "group":
                reply = await _reset_session_data(session_id)
            else:
                reply = "当前不是群聊消息，无法重置群会话"

        elif sub2.startswith("group:") or sub2.startswith("group："):
            sep = "：" if "：" in sub2 else ":"
            gid = sub2.split(sep, 1)[1].strip()
            if gid:
                reply = await _reset_session_data(f"group_{gid}")
            else:
                reply = "格式错误，用法: /admin 重置会话 group:<群号>"

        elif sub2.startswith("private:") or sub2.startswith("private："):
            sep = "：" if "：" in sub2 else ":"
            qq = sub2.split(sep, 1)[1].strip()
            if qq:
                reply = await _reset_session_data(f"private_{qq}")
            else:
                reply = "格式错误，用法: /admin 重置会话 private:<QQ号>"

        elif sub2 == "":
            if message_type == "group":
                desc = f"群聊{group_id}"
                pending_confirms[session_id] = {
                    "target": session_id, "time": time.time(), "desc": desc
                }
                reply = (
                    f"⚠️ 确定要重置{desc}的全部数据吗？\n"
                    f"包括情绪、好感度、认知资源、短期/长期记忆、情感锚点。\n"
                    f"如果是，请在10秒内发送 /admin 确认重置 来执行。"
                )
            else:
                reply = "私聊请明确子命令: /admin 重置会话 private"

        else:
            reply = (f"未知参数: {sub2}\n"
                     f"用法: /admin 重置会话 [private/group/group:<群号>/private:<QQ号>]")

    # ==================== 确认重置 ====================
    elif sub in ("确认重置", "confirmreset"):
        confirm = pending_confirms.pop(session_id, None)
        if confirm is None:
            reply = "当前没有待确认的重置操作"
        elif time.time() - confirm["time"] > 10:
            reply = "⏰ 确认超时（超过10秒），请重新发送 /admin 重置会话"
        else:
            reply = await _reset_session_data(confirm["target"])

    elif sub in ("取消重置", "cancelreset"):
        if pending_confirms.pop(session_id, None):
            reply = "已取消重置操作"
        else:
            reply = "当前没有待确认的重置操作"

    # ==================== 查看用户画像 ====================
    elif sub in ("画像", "profile"):
        sub2 = parts[2] if len(parts) > 2 else ""
        target_qq = None
        if sub2:
            try:
                target_qq = int(sub2)
            except ValueError:
                reply = f"参数错误: '{sub2}' 不是有效的 QQ 号"
                return await _send_admin_reply(reply, message_type, user_id, group_id, send_url, headers)
        elif message_type == "private":
            target_qq = user_id
        else:
            reply = "群聊请指定 QQ 号: /admin 画像 <QQ号>"
            return await _send_admin_reply(reply, message_type, user_id, group_id, send_url, headers)

        profile = await get_profile(target_qq)
        if not profile or not any(profile.values()):
            reply = f"用户 {target_qq} 没有画像数据"
        else:
            import json
            reply = f"用户 {target_qq} 画像:\n{json.dumps(profile, ensure_ascii=False, indent=2)}"

    # ==================== 重置用户画像 ====================
    elif sub in ("重置画像", "resetprofile"):
        sub2 = parts[2] if len(parts) > 2 else ""
        if sub2:
            # 指定了 QQ 号
            try:
                target_qq = int(sub2)
                deleted = await delete_profile(target_qq)
                if deleted:
                    reply = f"用户 {target_qq} 的画像已删除"
                else:
                    reply = f"用户 {target_qq} 没有画像数据"
            except ValueError:
                reply = f"参数错误: '{sub2}' 不是有效的 QQ 号"
        elif message_type == "private":
            # 私聊未指定 QQ 号 → 重置当前用户
            deleted = await delete_profile(user_id)
            if deleted:
                reply = f"当前用户的画像已删除"
            else:
                reply = f"当前用户没有画像数据"
        else:
            reply = "群聊请指定 QQ 号: /admin 重置画像 <QQ号>"

    # ==================== 帮助 ====================
    elif sub in ("帮助", "help"):
        reply = (
            "==== 管理员指令 ====\n"
            "/admin 资源 [子命令] — 资源系统（状态/set/rate/重置/爆发/exitburst）\n"
            "/admin 好感 [值/QQ号/群昵称] — 查看/设置自己的好感度，或查询他人好感度\n"
            "/admin 情绪 <名> <值> — 查看/设置情绪值\n"
            "/admin 清情绪 — 清空所有情绪\n"
            "/admin 检索 <词> — 检索长期记忆\n"
            "/admin 锚点 — 查看情感锚点\n"
            "/admin 清记忆 — 清空短期记忆\n"
            "/admin 上下文 [条数] — 查看聊天记录\n"
            "/admin prompt <文本> — 调试情绪分析\n"
            "/admin 重置会话 [子命令] — 重置会话数据\n"
            "/admin 确认重置 — 确认之前的重置操作\n"
            "/admin 取消重置 — 取消待确认的重置\n"
            "/admin 画像 [QQ号] — 查看用户画像（私聊留空=自己）\n"
            "/admin 重置画像 [QQ号] — 删除用户画像（私聊留空=自己）"
        )

    else:
        reply = f"未知子命令: {sub}。输入 /admin 帮助 查看可用指令"

    return await _send_admin_reply(reply, message_type, user_id, group_id, send_url, headers)


def _apply_social_filter_local(emotion_dict, reasons_dict, user_emotions, state):
    """包裹 emotion.social_filter.apply_social_filter 的本地引用"""
    from emotion.social_filter import apply_social_filter
    return apply_social_filter(emotion_dict, reasons_dict, user_emotions, state)
