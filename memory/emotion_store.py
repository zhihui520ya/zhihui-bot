"""
情绪 + 好感度 + 锚点标签 存储管理

所有写操作受会话级 asyncio.Lock 保护，防止并发读-改-写覆盖。
"""
import sqlite3
import datetime
import json
import os
import math
import asyncio
import time

# 会话级锁，防止同一 session 的并发读-改-写覆盖
_session_locks: dict[str, asyncio.Lock] = {}
_session_locks_lock = asyncio.Lock()


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    async with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]


def _get_conn():
    db_dir = os.path.join(DATA_DIR, "memories")
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(db_dir, "memory.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    # 创建情绪向量表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_emotions (
            session_id TEXT PRIMARY KEY,
            emotion_vector TEXT,
            last_updated TEXT
        )
    """)
    # 创建好感度表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_affection (
            session_id TEXT,
            user_id INTEGER,
            affection REAL DEFAULT 0,
            last_updated TEXT,
            PRIMARY KEY (session_id, user_id)
        )
    """)
    # 兼容旧表结构
    try:
        conn.execute("ALTER TABLE session_emotions ADD COLUMN emotion_vector TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE session_emotions ADD COLUMN reasons TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE session_emotions ADD COLUMN last_message_time TEXT")
    except sqlite3.OperationalError:
        pass
    # anchored_labels：记录哪些情绪标签曾达到过 ≥7（JSON数组）
    try:
        conn.execute("ALTER TABLE session_emotions ADD COLUMN anchored_labels TEXT")
    except sqlite3.OperationalError:
        pass
    # 兼容旧 user_affection 表：添加 last_updated 列
    try:
        conn.execute("ALTER TABLE user_affection ADD COLUMN last_updated TEXT")
    except sqlite3.OperationalError:
        pass
    # 情感锚点复活日志（持久化冷却，重启不丢失）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anchor_resurrect_log (
            session_id TEXT,
            anchor_id TEXT,
            resurrected_at REAL,
            PRIMARY KEY (session_id, anchor_id)
        )
    """)
    return conn


# ========== 内部无锁版本（供持锁上下文调用） ==========

def _get_session_emotion_no_lock(session_id, half_life_dict=None, baseline_dict=None):
    """
    无锁版：读取情绪向量 + 半衰期衰减 + 基底保底。
    调用方必须持有 _get_session_lock(session_id) 的保护。
    """
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT emotion_vector, last_updated, reasons FROM session_emotions WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {}, None, {}
    vec_str, last_updated_str, reasons_str = row
    last_updated = datetime.datetime.fromisoformat(last_updated_str)
    now = datetime.datetime.utcnow()
    delta_seconds = (now - last_updated).total_seconds()
    emotion_dict = json.loads(vec_str) if vec_str else {}
    reasons_dict = json.loads(reasons_str) if reasons_str else {}

    if delta_seconds <= 0:
        conn.close()
        return emotion_dict, last_updated, reasons_dict

    # 半衰期衰减
    if half_life_dict:
        baseline_dict = baseline_dict or {}
        for k in list(emotion_dict.keys()):
            half_life = half_life_dict.get(k, 3600)
            factor = 0.5 ** (delta_seconds / half_life)
            new_val = emotion_dict[k] * factor
            if new_val > 0.01:
                emotion_dict[k] = round(new_val, 2)
            else:
                del emotion_dict[k]

        # 基底保底
        for k, bv in baseline_dict.items():
            emotion_dict[k] = max(emotion_dict.get(k, 0), bv)

        # 更新 DB
        conn.execute(
            "UPDATE session_emotions SET emotion_vector = ?, last_updated = ? WHERE session_id = ?",
            (json.dumps(emotion_dict), now.isoformat(), session_id)
        )
        conn.commit()
        conn.close()
        return emotion_dict, last_updated, reasons_dict

    conn.close()
    return emotion_dict, last_updated, reasons_dict


def _set_session_emotion_no_lock(session_id, emotion_dict, reasons_dict=None, last_message_time=None):
    """无锁版：写入情绪向量。调用方必须持有会话锁。"""
    conn = _get_conn()
    reasons_json = json.dumps(reasons_dict) if reasons_dict else "{}"
    now_str = datetime.datetime.utcnow().isoformat()

    if last_message_time is not None:
        conn.execute(
            "INSERT OR REPLACE INTO session_emotions (session_id, emotion_vector, last_updated, reasons, last_message_time) VALUES (?, ?, ?, ?, ?)",
            (session_id, json.dumps(emotion_dict), now_str, reasons_json, last_message_time)
        )
    else:
        # 保留现有的 last_message_time，避免被 REPLACE 置空
        existing = conn.execute(
            "SELECT last_message_time FROM session_emotions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        existing_lmt = existing[0] if existing else None
        conn.execute(
            "INSERT OR REPLACE INTO session_emotions (session_id, emotion_vector, last_updated, reasons, last_message_time) VALUES (?, ?, ?, ?, ?)",
            (session_id, json.dumps(emotion_dict), now_str, reasons_json, existing_lmt)
        )
    conn.commit()
    conn.close()


def _record_anchor_label_no_lock(session_id, emotion_label):
    """无锁版：记录情绪锚点标签。调用方必须持有会话锁。"""
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT anchored_labels FROM session_emotions WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    labels = set()
    if row and row[0]:
        labels = set(json.loads(row[0]))
    labels.add(emotion_label)
    if row:
        conn.execute(
            "UPDATE session_emotions SET anchored_labels = ? WHERE session_id = ?",
            (json.dumps(list(labels)), session_id)
        )
    else:
        conn.execute(
            "INSERT INTO session_emotions (session_id, emotion_vector, last_updated, anchored_labels) VALUES (?, ?, ?, ?)",
            (session_id, "{}", datetime.datetime.utcnow().isoformat(), json.dumps(list(labels)))
        )
    conn.commit()
    conn.close()


# ========== 情绪向量（公开 API，带锁） ==========

async def get_session_emotion(session_id, half_life_dict=None, baseline_dict=None):
    """
    获取会话的情绪向量，自动衰减。

    返回 (emotion_dict, last_updated, reasons_dict)
    """
    async with await _get_session_lock(session_id):
        return _get_session_emotion_no_lock(session_id, half_life_dict, baseline_dict)


async def set_session_emotion(session_id, emotion_dict, reasons_dict=None, last_message_time=None):
    """
    设置会话的情绪向量（覆盖）。
    可选参数：reasons_dict, last_message_time
    """
    async with await _get_session_lock(session_id):
        _set_session_emotion_no_lock(session_id, emotion_dict, reasons_dict, last_message_time)


async def update_session_emotion(session_id, emotion_name, delta, max_val=10, reason=None):
    """
    对某个情绪维度增加 delta，并限制最大强度。
    reason: 引起该情绪变化的原因（可选）

    自动维护 anchored_labels：如果更新后值 ≥7，将该标签加入锚点标签列表
    自动清理原因：如果值归零，同时清除对应原因
    """
    async with await _get_session_lock(session_id):
        cur_dict, _, reasons_dict = _get_session_emotion_no_lock(session_id)
        new_val = cur_dict.get(emotion_name, 0) + delta
        if new_val <= 0:
            cur_dict.pop(emotion_name, None)
            reasons_dict.pop(emotion_name, None)  # 归零同时清理原因
        else:
            cur_dict[emotion_name] = min(max_val, new_val)
            if reason:
                reasons_dict[emotion_name] = reason

        _set_session_emotion_no_lock(session_id, cur_dict, reasons_dict)

        # 检查是否需要记录锚点标签
        clamped_val = min(max_val, new_val)
        if clamped_val >= 7:
            _record_anchor_label_no_lock(session_id, emotion_name)


async def get_anchored_labels(session_id):
    """获取该会话所有有过锚点的情绪标签集合"""
    async with await _get_session_lock(session_id):
        conn = _get_conn()
        cursor = conn.execute(
            "SELECT anchored_labels FROM session_emotions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return set(json.loads(row[0]))
        return set()


async def record_anchor_label(session_id, emotion_label):
    """确保某情绪标签被记录为锚点（曾达到过 ≥7）"""
    async with await _get_session_lock(session_id):
        _record_anchor_label_no_lock(session_id, emotion_label)


async def has_anchor_label(session_id, emotion_label):
    """检查某标签是否曾有过锚点"""
    labels = await get_anchored_labels(session_id)
    return emotion_label in labels


# ========== 情感锚点复活冷却（持久化，重启不丢失） ==========

async def is_anchor_in_cooldown(session_id: str, anchor_id: str, cooldown_seconds: int = 3600) -> bool:
    """
    检查锚点是否在冷却期内。返回 True 表示还在冷却，不应复活。
    线程安全：单条 SELECT，无需跨会话锁。
    """
    try:
        conn = _get_conn()
        cursor = conn.execute(
            "SELECT resurrected_at FROM anchor_resurrect_log WHERE session_id = ? AND anchor_id = ?",
            (session_id, anchor_id)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            last_ts = row[0]
            if time.time() - last_ts < cooldown_seconds:
                return True
        return False
    except Exception as e:
        print(f"[锚点冷却] 查询失败: {e}")
        return True  # 查询失败时保守处理，跳过复活


async def record_anchor_resurrect(session_id: str, anchor_id: str):
    """记录锚点已复活（写入当前时间戳，持久化到 DB）"""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO anchor_resurrect_log (session_id, anchor_id, resurrected_at) VALUES (?, ?, ?)",
            (session_id, anchor_id, time.time())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[锚点复活] 记录失败: {e}")


async def clear_session_emotion(session_id):
    """清除会话的情绪状态"""
    async with await _get_session_lock(session_id):
        conn = _get_conn()
        conn.execute("DELETE FROM session_emotions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()


async def get_last_message_time(session_id):
    """获取上条消息时间"""
    async with await _get_session_lock(session_id):
        conn = _get_conn()
        cursor = conn.execute(
            "SELECT last_message_time FROM session_emotions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return datetime.datetime.fromisoformat(row[0])
        return None


async def update_last_message_time(session_id):
    """更新上条消息时间为现在"""
    async with await _get_session_lock(session_id):
        conn = _get_conn()
        now_str = datetime.datetime.utcnow().isoformat()
        cursor = conn.execute(
            "SELECT emotion_vector FROM session_emotions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            conn.execute(
                "UPDATE session_emotions SET last_message_time = ? WHERE session_id = ?",
                (now_str, session_id)
            )
        else:
            conn.execute(
                "INSERT INTO session_emotions (session_id, emotion_vector, last_updated, last_message_time) VALUES (?, ?, ?, ?)",
                (session_id, "{}", datetime.datetime.utcnow().isoformat(), now_str)
            )
        conn.commit()
        conn.close()


# ========== 用户好感度管理 ==========
from config import AFFECTION_HALF_LIFE, AFFECTION_MIN, AFFECTION_MAX, DATA_DIR


def init_affection_table():
    """初始化好感度表。同步函数，通常仅启动时调用一次。"""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_affection (
            session_id TEXT,
            user_id INTEGER,
            affection REAL DEFAULT 0,
            last_updated TEXT,
            PRIMARY KEY (session_id, user_id)
        )
    """)
    conn.commit()
    conn.close()


async def get_user_affection(session_id, user_id):
    """
    获取该会话中某个用户的好感度，自动应用 7 天半衰期衰减。
    返回 float，范围 [0, 10000]
    """
    async with await _get_session_lock(session_id):
        return _get_user_affection_no_lock(session_id, user_id)


def _get_user_affection_no_lock(session_id, user_id) -> float:
    """
    无锁版：读取好感度并应用半衰期衰减。
    调用方必须持有 _get_session_lock(session_id) 的保护。
    """
    conn = _get_conn()
    cur = conn.execute(
        "SELECT affection, last_updated FROM user_affection WHERE session_id = ? AND user_id = ?",
        (session_id, user_id)
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return 2000.0

    affection = float(row[0])
    last_updated_str = row[1]

    if last_updated_str:
        last_updated = datetime.datetime.fromisoformat(last_updated_str)
        now = datetime.datetime.utcnow()
        dt = (now - last_updated).total_seconds()
        if dt > 0:
            factor = 0.5 ** (dt / AFFECTION_HALF_LIFE)
            new_val = affection * factor
            if abs(new_val) < 0.01:
                new_val = 0.0
            now_str = now.isoformat()
            conn.execute(
                "UPDATE user_affection SET affection = ?, last_updated = ? WHERE session_id = ? AND user_id = ?",
                (new_val, now_str, session_id, user_id)
            )
            conn.commit()
            conn.close()
            return round(max(AFFECTION_MIN, min(AFFECTION_MAX, new_val)), 2)

    conn.close()
    return round(max(AFFECTION_MIN, min(AFFECTION_MAX, affection)), 2)


async def update_user_affection(session_id, user_id, delta):
    """
    增加或减少好感度（delta 可为浮点数）。
    先获取当前值（含时间衰减），叠加 delta，限制范围，写回。
    """
    async with await _get_session_lock(session_id):
        current = _get_user_affection_no_lock(session_id, user_id)
        new_val = current + delta
        new_val = max(AFFECTION_MIN, min(AFFECTION_MAX, new_val))
        now_str = datetime.datetime.utcnow().isoformat()

        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO user_affection (session_id, user_id, affection, last_updated) VALUES (?, ?, ?, ?)",
            (session_id, user_id, round(new_val, 2), now_str)
        )
        conn.commit()
        conn.close()
