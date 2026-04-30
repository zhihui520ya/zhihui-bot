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


# ========== LLM 提取 ==========

_PROFILE_EXTRACTION_PROMPT = """你是一个用户画像分析师。根据以下对话，提取或更新该用户的画像信息。

当前画像：{existing_profile}

用户消息：{user_msg}
AI 回复：{bot_reply}

分析用户消息中是否有用户主动透露的个人信息。
严格规则（非常重要）：
1. 只提取用户消息中明确说出的关于TA自己的信息，例如"我叫XX""我今年X岁""我喜欢XX"。
2. 不要从AI的回复推断用户的喜好。例如用户问"这是谁"，AI回答"这是银狼"——这不意味着用户喜欢银狼。
3. 不要从聊天上下文推断用户特征。用户只是提了一个问题或发了一张图，不等于TA喜欢图中内容。
4. 除非用户明确说了"我喜欢XX""我讨厌XX""我的XX是XX"，否则视为没有新信息。
5. 如果用户消息只是一般提问、打招呼、发图、回复表情等日常对话，没有透露个人信息，返回 {"updated": false}。

可能的画像字段（不限于这些）：
- name: 用户称呼/昵称
- gender: 性别
- age: 年龄
- relationship: 与 AI 的关系描述
- likes: 喜欢的事物（列表，仅当用户明确说自己喜欢时记录）
- dislikes: 不喜欢的事物（列表，仅当用户明确说自己不喜欢时记录）
- personality: 性格特征描述
- facts: 其他事实（列表，仅当用户陈述关于自己的事实时记录）
- preferences: 偏好（对象，仅当用户明确表达偏好时记录）

返回格式：
{{"updated": true, "profile": {{"name": "小明", "age": "18", ...}}}}
或
{{"updated": false}}

只返回 JSON，不要有其他文字。"""


async def extract_and_update(user_id: int, user_msg: str, bot_reply: str, llm: ChatOpenAI):
    """后台调用 LLM 提取用户画像信息并更新。"""
    if not user_id or not user_msg:
        return

    async with await _get_user_lock(user_id):
        existing = _get_profile_no_lock(user_id)

    existing_json = json.dumps(existing, ensure_ascii=False) if existing else "{}"
    prompt = _PROFILE_EXTRACTION_PROMPT.format(
        existing_profile=existing_json,
        user_msg=user_msg[:500],
        bot_reply=bot_reply[:500],
    )

    try:
        response = await llm.ainvoke(prompt)
        result_text = response.content.strip()
        # 清理可能的 markdown 代码块标记
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[-1]
            result_text = result_text.rsplit("```", 1)[0]
        result = json.loads(result_text.strip())

        if result.get("updated") and "profile" in result:
            updates = result["profile"]
            if isinstance(updates, dict) and updates:
                async with await _get_user_lock(user_id):
                    _merge_profile_no_lock(user_id, updates)
                print(f"[画像] 提取并更新: user_id={user_id}, 字段={list(updates.keys())}")
    except json.JSONDecodeError:
        print(f"[画像] LLM 返回非 JSON，跳过提取: {result_text[:100]}")
    except Exception as e:
        print(f"[画像] 提取失败: {e}")


# 兼容性别名
UserProfileManager = None  # 保持模块级调用风格，无需实例化
