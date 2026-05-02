"""
用户画像管理 —— 跨会话（私聊/群聊）用户档案。

以 QQ user_id 为主键，不受 session_id（private_/group_）限制。
所有写操作受会话级 asyncio.Lock 保护。
"""
import sqlite3
import json
import os
import asyncio
import datetime
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from config import DATA_DIR

_session_locks: dict[int, asyncio.Lock] = {}
_session_locks_guard = asyncio.Lock()


async def _get_user_lock(user_id: int) -> asyncio.Lock:
    async with _session_locks_guard:
        if user_id not in _session_locks:
            _session_locks[user_id] = asyncio.Lock()
        return _session_locks[user_id]


def _get_conn():
    db_dir = os.path.join(DATA_DIR, "memories")
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(db_dir, "user_profiles.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            profile_json TEXT NOT NULL DEFAULT '{}',
            message_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_message_buffer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_msg TEXT NOT NULL,
            bot_reply TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_buffer_user_id ON user_message_buffer(user_id)
    """)
    return conn


# ========== 无锁版本（供持锁上下文调用） ==========

def _get_profile_no_lock(user_id: int) -> dict:
    """无锁版：读取用户画像。调用方必须持有 _get_user_lock(user_id) 的保护。"""
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT profile_json FROM user_profiles WHERE user_id = ?",
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return json.loads(row[0])
    return {}


def _upsert_profile_no_lock(user_id: int, profile: dict, increment_count: bool = False):
    """无锁版：写入用户画像（覆盖）。调用方必须持有 _get_user_lock(user_id) 的保护。"""
    conn = _get_conn()
    now_str = datetime.datetime.utcnow().isoformat()
    existing = conn.execute(
        "SELECT profile_json, message_count FROM user_profiles WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if existing:
        new_count = existing[1] + 1 if increment_count else existing[1]
        conn.execute(
            "UPDATE user_profiles SET profile_json = ?, message_count = ?, updated_at = ? WHERE user_id = ?",
            (json.dumps(profile, ensure_ascii=False), new_count, now_str, user_id)
        )
    else:
        conn.execute(
            "INSERT INTO user_profiles (user_id, profile_json, message_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, json.dumps(profile, ensure_ascii=False), 1, now_str, now_str)
        )
    conn.commit()
    conn.close()


def _merge_profile_no_lock(user_id: int, updates: dict, increment_count: bool = False):
    """无锁版：合并更新。调用方必须持有 _get_user_lock(user_id) 的保护。"""
    current = _get_profile_no_lock(user_id)
    _deep_merge(current, updates)
    _upsert_profile_no_lock(user_id, current, increment_count)


def _deep_merge(base: dict, updates: dict):
    """递归合并两个 dict，list 字段直接替换，不追加。"""
    for k, v in updates.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ========== 公开 API（带锁） ==========

async def get_profile(user_id: int) -> dict:
    """获取用户画像，返回 dict。从不在该用户发言时返回空 dict。"""
    async with await _get_user_lock(user_id):
        return _get_profile_no_lock(user_id)


async def update_profile(user_id: int, updates: dict):
    """合并更新用户画像。"""
    async with await _get_user_lock(user_id):
        _merge_profile_no_lock(user_id, updates)


async def delete_profile(user_id: int) -> bool:
    """删除指定用户的画像。返回 True 表示存在并被删除，False 表示不存在。"""
    async with await _get_user_lock(user_id):
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        if deleted:
            print(f"[画像] 删除: user_id={user_id}")
    return deleted


async def record_message(user_id: int):
    """记录一次用户发言（增加 message_count）。"""
    async with await _get_user_lock(user_id):
        conn = _get_conn()
        now_str = datetime.datetime.utcnow().isoformat()
        existing = conn.execute(
            "SELECT message_count FROM user_profiles WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE user_profiles SET message_count = ?, updated_at = ? WHERE user_id = ?",
                (existing[0] + 1, now_str, user_id)
            )
        else:
            conn.execute(
                "INSERT INTO user_profiles (user_id, profile_json, message_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, "{}", 1, now_str, now_str)
            )
        conn.commit()
        conn.close()


# ========== 消息缓存（批量分析用） ==========

# 每用户缓存多少轮对话后自动触发批量分析
BUFFER_FLUSH_THRESHOLD = 20


async def add_exchange_to_buffer(user_id: int, user_msg: str, bot_reply: str) -> int:
    """缓存一轮对话到缓冲区。返回当前缓冲区大小（该用户的总缓存条数）。"""
    now_str = datetime.datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO user_message_buffer (user_id, user_msg, bot_reply, created_at) VALUES (?, ?, ?, ?)",
        (user_id, user_msg, bot_reply, now_str)
    )
    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) FROM user_message_buffer WHERE user_id = ?",
        (user_id,)
    ).fetchone()[0]
    conn.close()
    return count


async def get_buffer_size(user_id: int) -> int:
    """获取该用户的缓冲区消息条数。"""
    conn = _get_conn()
    count = conn.execute(
        "SELECT COUNT(*) FROM user_message_buffer WHERE user_id = ?",
        (user_id,)
    ).fetchone()[0]
    conn.close()
    return count


async def flush_and_analyze(user_id: int, llm: ChatOpenAI) -> bool:
    """清空缓冲区并用所有缓存的对话批量分析用户画像。返回 True 表示画像有更新。"""
    async with await _get_user_lock(user_id):
        conn = _get_conn()
        rows = conn.execute(
            "SELECT user_msg, bot_reply FROM user_message_buffer WHERE user_id = ? ORDER BY id",
            (user_id,)
        ).fetchall()
        if not rows:
            conn.close()
            return False

        # 构建完整对话文本
        exchanges = []
        for row in rows:
            exchanges.append(f"用户: {row[0]}\n知慧: {row[1]}")
        full_text = "\n\n---\n\n".join(exchanges)

        # 清空缓冲区
        conn.execute("DELETE FROM user_message_buffer WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()

    # 读取当前画像
    existing = await get_profile(user_id)
    existing_json = json.dumps(existing, ensure_ascii=False) if existing else "{}"

    prompt = _BATCH_PROFILE_EXTRACTION_PROMPT.format(
        existing_profile=existing_json,
        exchanges=full_text,
    )

    try:
        response = await llm.ainvoke(prompt)
        result_text = response.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[-1]
            result_text = result_text.rsplit("```", 1)[0]
        result = json.loads(result_text.strip())

        if isinstance(result, dict) and result.get("updated"):
            updates = {}
            if "profile" in result:
                updates.update(result["profile"])
            if "_observations" in result:
                updates["_observations"] = result["_observations"]

            if updates:
                async with await _get_user_lock(user_id):
                    _merge_profile_no_lock(user_id, updates)
                print(f"[画像] 批量分析更新: user_id={user_id}, 字段={list(updates.keys())}")
                return True
    except Exception as e:
        print(f"[画像] 批量分析失败: {e}")

    return False


# ========== LLM 提取（单轮/兼容） ==========

_PROFILE_EXTRACTION_PROMPT = """你是一个用户画像分析师。根据以下对话，提取或更新该用户的画像信息。

当前画像：{existing_profile}

用户消息：{user_msg}
AI 回复：{bot_reply}

分析用户消息并提取信息。

=== 个人信息（严格模式） ===
只提取用户明确说的关于TA自己的事实：
- name: 称呼/昵称
- gender: 性别
- age: 年龄
- likes: 喜欢的事物（列表）
- dislikes: 不喜欢的事物（列表）
- personality: 性格特征
- facts: 其他事实（列表）

=== 行为观察（宽松模式） ===
根据用户怎么说话，推断行为模式：
- communication_style: 沟通风格（直接/委婉/幽默/简洁/话痨/理性/感性）
- emotional_pattern: 情绪模式（稳定/敏感/热情/冷静/易怒/乐观）
- attitude_to_bot: 对 bot 的态度（友好/随意/挑剔/依赖/客气/亲近）
- typical_topics: 常聊话题（列表）
- interaction_preference: 互动偏好（深度聊天/轻松闲聊/快速问答/工具使用）

规则：
- 个人信息：只提取用户明确说出的，没有就留空
- 行为观察：根据整段对话的语气和风格推断，没有明显特征就留空
- 只有本轮对话有新信息时才返回 {"updated": true}

返回格式：
{{"updated": true, "profile": {{"name": "小明", "age": "18", "communication_style": "直接幽默"}}}}
或
{{"updated": false}}

只返回 JSON，不要有其他文字。"""

_BATCH_PROFILE_EXTRACTION_PROMPT = """你是一个用户画像分析师。根据以下多轮对话，提取或更新该用户的画像信息。

当前画像：{existing_profile}

以下是该用户的多轮对话记录：
{exchanges}

分析用户在所有这些对话中透露的信息。

=== 个人信息（严格模式） ===
只提取用户明确说的关于TA自己的事实：
- name: 称呼/昵称
- gender: 性别
- age: 年龄
- likes: 喜欢的事物（列表）
- dislikes: 不喜欢的事物（列表）
- personality: 性格特征
- facts: 其他事实（列表）

=== 行为观察（宽松模式） ===
根据用户怎么说话，推断行为模式：
- communication_style: 沟通风格（直接/委婉/幽默/简洁/话痨/理性/感性）
- emotional_pattern: 情绪模式（稳定/敏感/热情/冷静/易怒/乐观）
- attitude_to_bot: 对 bot 的态度（友好/随意/挑剔/依赖/客气/亲近）
- typical_topics: 常聊话题（列表）
- interaction_preference: 互动偏好（深度聊天/轻松闲聊/快速问答/工具使用）

规则：
- 个人信息：只提取用户明确说出的，没有就留空
- 行为观察：根据所有对话的语气和风格推断，没有明显特征就留空
- 结合新对话和当前画像，输出合并后的完整画像
- 行为观察单独放在 _observations 字段中

返回格式：
{{"updated": true, "profile": {{"name": "小明", "age": "18"}}, "_observations": {{"communication_style": "直接幽默", "typical_topics": ["游戏", "音乐"]}}}}
或
{{"updated": false}}

只返回 JSON，不要有其他文字。"""


async def extract_and_update(user_id: int, user_msg: str, bot_reply: str, llm: ChatOpenAI):
    """后台调用 LLM 提取用户画像信息并更新。"""
    if not user_id or not user_msg:
        return

    try:
        async with await _get_user_lock(user_id):
            existing = _get_profile_no_lock(user_id)

        existing_json = json.dumps(existing, ensure_ascii=False) if existing else "{}"
        prompt = _PROFILE_EXTRACTION_PROMPT.format(
            existing_profile=existing_json,
            user_msg=user_msg[:500],
            bot_reply=bot_reply[:500],
        )

        response = await llm.ainvoke(prompt)
        result_text = response.content.strip()
        # 清理可能的 markdown 代码块标记
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[-1]
            result_text = result_text.rsplit("```", 1)[0]
        result = json.loads(result_text.strip())

        if isinstance(result, dict) and result.get("updated") and "profile" in result:
            updates = result["profile"]
            if isinstance(updates, dict) and updates:
                async with await _get_user_lock(user_id):
                    _merge_profile_no_lock(user_id, updates)
                print(f"[画像] 提取并更新: user_id={user_id}, 字段={list(updates.keys())}")
    except json.JSONDecodeError:
        print(f"[画像] LLM 返回非 JSON，跳过提取")
    except (KeyError, AttributeError, TypeError) as _pe:
        import traceback
        print(f"[画像] 解析结果异常: {_pe}")
        print(f"[画像] Traceback:\n{traceback.format_exc()}")
    except Exception as e:
        print(f"[画像] 提取失败: {e}")


# 兼容性别名
UserProfileManager = None  # 保持模块级调用风格，无需实例化
