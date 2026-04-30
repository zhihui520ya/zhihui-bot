"""
统一的会话管理 API —— 提供 reset_session 用于彻底清理某会话的所有持久化数据。

调用方式：
    from api import reset_session
    cleaned = await reset_session(session_id)
"""
import os
import json
import sqlite3
from memory.emotion_store import clear_session_emotion
from emotion.social_filter import CognitiveResourceManager
from memory.memory_manager import ShortTermMemory
from memory.chroma_store import get_vector_store
from config import DATA_DIR

MEMORIES_DIR = os.path.join(DATA_DIR, "memories")
MAX_PROCESSED_FILE = os.path.join(MEMORIES_DIR, "max_processed.json")


async def reset_session(session_id: str) -> list[str]:
    """
    重置指定会话的所有持久化数据，返回已清理的组件名列表。
    涵盖：情绪向量、好感度、认知资源文件、短期记忆 DB、ChromaDB 向量集合、整理进度。
    """
    cleaned = []

    # 1. 情绪向量（memory.db - session_emotions 表）
    try:
        await clear_session_emotion(session_id)
        cleaned.append("情绪")
    except Exception as e:
        print(f"[reset] 清除情绪失败: {e}")

    # 2. 好感度（memory.db - user_affection 表）+ 锚点复活日志
    try:
        conn = sqlite3.connect(os.path.join(MEMORIES_DIR, "memory.db"))
        conn.execute(
            "DELETE FROM user_affection WHERE session_id = ?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM anchor_resurrect_log WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
        conn.close()
        cleaned.append("好感度")
    except Exception as e:
        print(f"[reset] 清除好感度失败: {e}")

    # 3. 认知资源文件（memories/cognitive_resources/{session_id}.json）
    res_path = os.path.join(MEMORIES_DIR, "cognitive_resources", f"{session_id}.json")
    try:
        if os.path.exists(res_path):
            os.remove(res_path)
        # 同时清除内存中的缓存实例
        CognitiveResourceManager._instances.pop(session_id, None)
        cleaned.append("认知资源")
    except Exception as e:
        print(f"[reset] 清除认知资源失败: {e}")

    # 4. 短期记忆 DB（memories/{session_id}.db）
    db_path = os.path.join(MEMORIES_DIR, f"{session_id}.db")
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        else:
            # DB 可能不存在但内存中有数据，尝试 clear
            ShortTermMemory(session_id).clear()
        cleaned.append("短期记忆")
    except Exception as e:
        print(f"[reset] 清除短期记忆失败: {e}")

    # 5. 长期记忆 / 情感锚点（ChromaDB 集合）
    try:
        vs = get_vector_store()
        await vs.delete_collection(session_id)
        cleaned.append("长期记忆/锚点")
    except Exception as e:
        print(f"[reset] 清除向量库失败: {e}")

    # 6. 记忆整理进度（max_processed.json）
    try:
        if os.path.exists(MAX_PROCESSED_FILE):
            with open(MAX_PROCESSED_FILE) as f:
                data = json.load(f)
            if session_id in data:
                del data[session_id]
                with open(MAX_PROCESSED_FILE, "w") as f:
                    json.dump(data, f)
                cleaned.append("记忆整理进度")
    except Exception as e:
        print(f"[reset] 清除整理进度失败: {e}")

    return cleaned
