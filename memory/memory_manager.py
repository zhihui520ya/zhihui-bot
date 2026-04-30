# memory_manager.py
import os
import json
import time
import asyncio
import datetime
import sqlite3
import shutil
import threading
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from memory import get_vector_store  # 现有向量库
from config import DATA_DIR

# ========== 配置常量 ==========
SHORT_TERM_MAX_ROUNDS = 50          # 最大轮数（AI 回复数）
SHORT_TERM_MIN_ROUNDS = 30          # 最小轮数阈值（只有达到此轮数才总结）
SHORT_TERM_CACHE_DIR = os.path.join(DATA_DIR, "memories", "short_term_cache")
LONG_TERM_MAX_PROCESSED_FILE = os.path.join(DATA_DIR, "memories", "max_processed.json")

# ========== 短期记忆类 ==========
class ShortTermMemory:
    """管理短期库（SQLite），存储最近 N 轮对话，支持总结与清空"""
    MAX_ROUNDS = SHORT_TERM_MAX_ROUNDS
    MIN_ROUNDS = SHORT_TERM_MIN_ROUNDS

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.db_path = os.path.join(DATA_DIR, "memories", f"{session_id}.db")
        self.lock = threading.Lock()
        self._ensure_db()

    def _ensure_db(self):
        """确保数据库和表存在"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    message TEXT,
                    napcat_msg_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 迁移：为旧数据库添加 napcat_msg_id 列（如果缺失）
            try:
                conn.execute("ALTER TABLE message_store ADD COLUMN napcat_msg_id INTEGER")
            except sqlite3.OperationalError:
                pass  # 列已存在
            conn.commit()
            conn.close()

    def update_message_content(self, napcat_msg_id: int, new_content: str) -> bool:
        """根据 NapCat message_id 更新已存储消息的内容"""
        msg_dict = {
            "type": "human",
            "data": {
                "content": new_content,
                "additional_kwargs": {"user_id": ""},
                "example": False,
            },
        }
        # 先读出现有内容，保留 user_id
        row = self._execute(
            "SELECT message FROM message_store WHERE napcat_msg_id = ?",
            (napcat_msg_id,), fetchone=True
        )
        if not row:
            return False
        try:
            old_data = json.loads(row["message"])
            uid = old_data.get("data", {}).get("additional_kwargs", {}).get("user_id", "")
            msg_dict["data"]["additional_kwargs"]["user_id"] = uid
        except Exception:
            pass
        msg_json = json.dumps(msg_dict, ensure_ascii=False)
        self._execute(
            "UPDATE message_store SET message = ? WHERE napcat_msg_id = ?",
            (msg_json, napcat_msg_id)
        )
        print(f"[记忆] 更新消息 napcat_msg_id={napcat_msg_id}: {new_content[:60]}...")
        return True

    def delete_by_napcat_msg_id(self, napcat_msg_id: int) -> bool:
        """根据 NapCat message_id 删除一条消息（撤回时使用）"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.execute(
                "DELETE FROM message_store WHERE napcat_msg_id = ?",
                (napcat_msg_id,),
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            if deleted:
                print(f"[记忆] 已删除撤回消息 napcat_msg_id={napcat_msg_id}")
            return bool(deleted)

    def _execute(self, sql, params=(), fetchone=False, fetchall=False):
        """执行 SQL，自动加锁"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            if fetchone:
                result = cursor.fetchone()
            elif fetchall:
                result = cursor.fetchall()
            else:
                result = None
            conn.commit()
            conn.close()
            return result

    def add_message(self, role: str, content: str, timestamp=None, user_id=None, napcat_msg_id=None):
        """添加一条消息，role 为 'human' 或 'ai'

        Args:
            role: 'human' 或 'ai'
            content: 消息内容
            timestamp: 可选的时间戳（datetime对象或Unix时间戳），默认为当前时间
            user_id: 发送者QQ号（仅 human 消息需要区分谁说的）
            napcat_msg_id: NapCat 消息ID（用于撤回时查找删除）
        """
        additional_kwargs = {}
        if role == 'human' and user_id:
            additional_kwargs['user_id'] = str(user_id)

        msg_dict = {
            "type": role,
            "data": {
                "content": content,
                "additional_kwargs": additional_kwargs,
                "example": False
            }
        }
        msg_json = json.dumps(msg_dict, ensure_ascii=False)

        if timestamp is not None:
            if isinstance(timestamp, (int, float)):
                # Unix时间戳转换为datetime
                dt = datetime.datetime.fromtimestamp(timestamp)
                created_at = dt.isoformat()
            elif isinstance(timestamp, datetime.datetime):
                created_at = timestamp.isoformat()
            else:
                created_at = timestamp
            self._execute(
                "INSERT INTO message_store (session_id, message, napcat_msg_id, created_at) VALUES (?, ?, ?, ?)",
                (self.session_id, msg_json, napcat_msg_id, created_at)
            )
        else:
            self._execute(
                "INSERT INTO message_store (session_id, message, napcat_msg_id) VALUES (?, ?, ?)",
                (self.session_id, msg_json, napcat_msg_id)
            )

    def _format_time_tag(self, created_at) -> str:
        """将 created_at 格式化为时间标签，如 [04-30 08:00]"""
        if not created_at:
            return ""
        try:
            if isinstance(created_at, str):
                dt = datetime.datetime.fromisoformat(created_at)
            elif isinstance(created_at, (int, float)):
                dt = datetime.datetime.fromtimestamp(created_at)
            else:
                return ""
            return dt.strftime("[%m-%d %H:%M] ")
        except Exception:
            return ""

    def get_recent_messages(self, k: int = 50) -> List[BaseMessage]:
        """获取最近 k 条消息（按时间正序），每条附带时间标签"""
        rows = self._execute(
            "SELECT message, created_at FROM message_store ORDER BY id DESC LIMIT ?",
            (k,), fetchall=True
        )
        messages = []
        for row in rows:
            msg_data = json.loads(row['message'])
            content = msg_data['data']['content']
            additional_kwargs = msg_data['data'].get('additional_kwargs', {})
            time_tag = self._format_time_tag(row.get('created_at'))
            if msg_data['type'] == 'human':
                uid = additional_kwargs.get('user_id')
                if uid:
                    display_content = f"{time_tag}[{uid}] {content}"
                else:
                    display_content = f"{time_tag}{content}"
                messages.append(HumanMessage(content=display_content))
            elif msg_data['type'] == 'ai':
                messages.append(AIMessage(content=f"{time_tag}{content}"))
        # 返回顺序为正序（最早的在前）
        messages.reverse()
        return messages

    def get_total_rounds(self) -> int:
        """获取当前对话轮数（AI 消息数）"""
        rows = self._execute(
            "SELECT message FROM message_store ORDER BY id",
            fetchall=True
        )
        count = 0
        for row in rows:
            msg_data = json.loads(row['message'])
            if msg_data['type'] == 'ai':
                count += 1
        return count

    def get_total_messages(self) -> int:
        """获取总消息数（用户+AI）"""
        row = self._execute("SELECT COUNT(*) as cnt FROM message_store", fetchone=True)
        return row['cnt'] if row else 0

    def get_last_user_message(self):
        """获取最后一条用户消息（内容、时间），用于话题切换检测"""
        rows = self._execute(
            "SELECT message, created_at FROM message_store WHERE message LIKE '%\"type\": \"human\"%' ORDER BY id DESC LIMIT 1",
            fetchall=True
        )
        if not rows:
            return None, None
        row = rows[0]
        msg_data = json.loads(row['message'])
        content = msg_data['data']['content']
        created_at = row['created_at']
        if isinstance(created_at, str):
            dt = datetime.datetime.fromisoformat(created_at)
        else:
            dt = created_at
        return content, dt

    def clear(self):
        """清空短期库"""
        self._execute("DELETE FROM message_store")

    def export_to_file(self, timestamp: int = None):
        """导出当前数据库到缓存目录"""
        os.makedirs(SHORT_TERM_CACHE_DIR, exist_ok=True)
        if timestamp is None:
            timestamp = int(time.time())
        dest = os.path.join(SHORT_TERM_CACHE_DIR, f"{self.session_id}_{timestamp}.db")
        shutil.copy2(self.db_path, dest)
        return dest

    async def summarize_and_store(self, llm: ChatOpenAI):
        """调用 LLM 生成摘要，存入向量库，然后清空短期库"""
        rows = self._execute(
            "SELECT message, created_at FROM message_store ORDER BY id",
            fetchall=True
        )
        if not rows:
            return
        conversation_text = []
        for row in rows:
            msg_data = json.loads(row['message'])
            role = msg_data['type']
            content = msg_data['data']['content']
            created_at = row['created_at']
            if isinstance(created_at, str):
                dt = datetime.datetime.fromisoformat(created_at)
            else:
                dt = created_at
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            conversation_text.append(f"{role}: {content}")
        full_text = "\n".join(conversation_text)

        prompt = f"""请根据以下对话内容，生成一段简短的摘要（200字以内），概括主要话题和关键信息：
{full_text}

摘要："""
        try:
            response = await llm.ainvoke(prompt)
            summary = response.content.strip()
            vector_store = get_vector_store()
            max_id = await vector_store.get_max_id(self.session_id) or 0
            new_id = max_id + 1
            await vector_store.add_summary(self.session_id, new_id, summary, metadata={
                "type": "summary",
                "timestamp": int(time.time()),
                "source": "short_term_summary"
            })
            print(f"[记忆] 已生成摘要，编号 {new_id}，存入向量库")
        except Exception as e:
            print(f"[记忆] 总结失败: {e}")
        finally:
            self.clear()
            print(f"[记忆] 短期库已清空 (session: {self.session_id})")


# ========== 长期记忆管理（向量库整理） ==========
class LongTermMemory:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.lock = threading.Lock()

    def get_max_processed(self, session_id: str) -> int:
        """获取最后一次整理的最大编号"""
        if os.path.exists(LONG_TERM_MAX_PROCESSED_FILE):
            with open(LONG_TERM_MAX_PROCESSED_FILE, 'r') as f:
                data = json.load(f)
                return data.get(session_id, 0)
        return 0

    def set_max_processed(self, session_id: str, value: int):
        """保存最大编号"""
        data = {}
        if os.path.exists(LONG_TERM_MAX_PROCESSED_FILE):
            with open(LONG_TERM_MAX_PROCESSED_FILE, 'r') as f:
                data = json.load(f)
        data[session_id] = value
        with open(LONG_TERM_MAX_PROCESSED_FILE, 'w') as f:
            json.dump(data, f)

    async def incremental_organize(self, session_id: str, llm: ChatOpenAI):
        """增量整理该会话的向量库段落"""
        max_processed = self.get_max_processed(session_id)
        all_ids = await self.vector_store.get_all_ids(session_id)
        if not all_ids:
            return
        # 过滤出编号大于 max_processed 的 id（id 是数字字符串）
        pending_ids = [id for id in all_ids if id.isdigit() and int(id) > max_processed]
        if not pending_ids:
            return
        pending_ids.sort(key=int)
        # 获取这些段落的内容和元数据
        paragraphs = await self.vector_store.get_by_ids(session_id, pending_ids)
        texts = []
        for para in paragraphs:
            texts.append(f"编号 {para['id']}: {para['document']}")
        full_text = "\n".join(texts)

        prompt = f"""你是一个记忆整理助手。以下是某个对话会话的多个记忆段落，每个段落有编号和内容。
请根据话题相似性将这些段落分组，每组生成一段新的总结（保留关键信息，避免信息丢失）。返回结果必须是 JSON 格式，示例如下：
[
  {{"start_id": 1, "end_id": 3, "summary": "新总结内容..."}},
  {{"start_id": 4, "end_id": 5, "summary": "新总结内容..."}}
]
注意：组必须覆盖所有段落，编号区间必须连续（如 1-3），且不能重叠。如果某个段落与其他都不同，可单独成组（start_id = end_id = 该编号）。

待整理段落：
{full_text}

请返回 JSON 列表："""
        try:
            response = await llm.ainvoke(prompt)
            result_text = response.content.strip()
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result_text, re.DOTALL)
            if json_match:
                groups = json.loads(json_match.group())
            else:
                raise ValueError("未找到 JSON")
            # 更新向量库
            for group in groups:
                start = group['start_id']
                end = group['end_id']
                summary = group['summary']
                ids_to_delete = [str(i) for i in range(start, end+1)]
                await self.vector_store.delete_by_ids(session_id, ids_to_delete)
                new_id = start
                await self.vector_store.add_summary(session_id, new_id, summary, metadata={
                    "type": "organized_summary",
                    "timestamp": int(time.time())
                })
            # 更新 max_processed 为本次处理的最大原始编号
            max_processed = max(pending_ids, key=int)
            self.set_max_processed(session_id, max_processed)
            print(f"[记忆] 增量整理完成，session: {session_id}, 处理到编号 {max_processed}")
        except Exception as e:
            print(f"[记忆] 增量整理失败: {e}")


# ========== 话题切换检测 ==========
def detect_topic_switch(short_term: ShortTermMemory, current_input: str,
                        time_threshold_minutes: int = 30,
                        similarity_threshold: float = 0.3) -> bool:
    """
    检测话题切换（基于时间间隔和 Jaccard 相似度）
    参数：
        short_term: ShortTermMemory 实例
        current_input: 当前用户输入
        time_threshold_minutes: 时间阈值（分钟）
        similarity_threshold: Jaccard 相似度阈值
    返回 True 表示话题切换
    """
    last_content, last_time = short_term.get_last_user_message()
    if last_content is None or last_time is None:
        return False
    now = datetime.datetime.now()
    time_diff = (now - last_time).total_seconds() / 60.0
    if time_diff > time_threshold_minutes:
        return True
    # 计算 Jaccard 相似度（基于字符集合）
    set1 = set(current_input)
    set2 = set(last_content)
    if not set1 and not set2:
        return False
    jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
    return jaccard < similarity_threshold