# memory_manager.py
import os
import json
import time
import asyncio
import datetime
import sqlite3
import shutil
import threading
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from config import DATA_DIR

# ========== 配置常量 ==========
SHORT_TERM_MAX_ROUNDS = 50          # 最大轮数（AI 回复数）
SHORT_TERM_MIN_ROUNDS = 50          # 最小轮数阈值（只有达到此轮数才总结）
SHORT_TERM_CACHE_DIR = os.path.join(DATA_DIR, "memories", "short_term_cache")

# ========== 短期记忆类 ==========
class ShortTermMemory:
    """管理短期库（SQLite），存储最近 N 轮对话，支持总结与清空"""
    MAX_ROUNDS = SHORT_TERM_MAX_ROUNDS
    MIN_ROUNDS = SHORT_TERM_MIN_ROUNDS

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.db_path = os.path.join(DATA_DIR, "memories", f"{session_id}.db")
        self.lock = threading.Lock()
        self._summarizing = False
        self._llm = None
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

    def set_llm(self, llm_instance):
        """注入 LLM 实例，用于自动轮数总结"""
        self._llm = llm_instance

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

        # 自动轮数总结触发（在锁释放后执行，不阻塞主流程）
        if self._llm is not None and not self._summarizing:
            rounds = self.get_total_rounds()
            if rounds >= SHORT_TERM_MIN_ROUNDS:
                self._summarizing = True
                path = self.export_to_file()
                self.clear()
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self._do_summarize(path))
                except RuntimeError:
                    pass

    def _format_time_tag(self, created_at) -> str:
        """将 created_at 格式化为时间后缀，如 （04-30 08:00）"""
        if not created_at:
            return ""
        try:
            if isinstance(created_at, str):
                dt = datetime.datetime.fromisoformat(created_at)
            elif isinstance(created_at, (int, float)):
                dt = datetime.datetime.fromtimestamp(created_at)
            else:
                return ""
            return dt.strftime("（%m-%d %H:%M）")
        except Exception:
            return ""

    def get_recent_messages(self, k: int = 30) -> List[BaseMessage]:
        """获取最近 k 条消息（按时间正序）。"""
        messages = []
        rows = self._execute(
            "SELECT message, created_at FROM message_store ORDER BY id DESC LIMIT ?",
            (k,), fetchall=True
        )
        for row in rows:
            msg_data = json.loads(row['message'])
            content = msg_data['data']['content']
            additional_kwargs = msg_data['data'].get('additional_kwargs', {})
            time_tag = self._format_time_tag(row['created_at'])
            if msg_data['type'] == 'human':
                uid = additional_kwargs.get('user_id')
                if uid:
                    display_content = f"[{uid}] {content} {time_tag}"
                else:
                    display_content = f"{content} {time_tag}"
                messages.append(HumanMessage(content=display_content))
            elif msg_data['type'] == 'ai':
                messages.append(AIMessage(content=f"{content} {time_tag}"))
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

    def delete_last_n(self, n: int = 1):
        """删除最近 N 条消息"""
        self._execute(
            "DELETE FROM message_store WHERE id IN (SELECT id FROM message_store ORDER BY id DESC LIMIT ?)",
            (n,),
        )

    def delete_by_user_id(self, user_id: str, limit: int | None = None) -> int:
        """删除指定用户的消息（根据 user_id）。返回删除条数。"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                if limit is not None:
                    cursor = conn.execute(
                        """DELETE FROM message_store WHERE id IN (
                            SELECT id FROM message_store
                            WHERE json_extract(message, '$.data.additional_kwargs.user_id') = ?
                            ORDER BY id DESC LIMIT ?
                        )""",
                        (user_id, limit),
                    )
                else:
                    cursor = conn.execute(
                        "DELETE FROM message_store WHERE json_extract(message, '$.data.additional_kwargs.user_id') = ?",
                        (user_id,),
                    )
                deleted = cursor.rowcount
                conn.commit()
            finally:
                conn.close()
            return deleted

    def export_to_file(self, timestamp: int = None):
        """导出当前数据库到缓存目录"""
        os.makedirs(SHORT_TERM_CACHE_DIR, exist_ok=True)
        if timestamp is None:
            timestamp = int(time.time())
        dest = os.path.join(SHORT_TERM_CACHE_DIR, f"{self.session_id}_{timestamp}.db")
        shutil.copy2(self.db_path, dest)
        return dest

    async def _do_summarize(self, snapshot_path: str):
        """从快照文件中读取对话，LLM 总结后存入向量库，删除快照。"""
        try:
            snap_conn = sqlite3.connect(snapshot_path, check_same_thread=False)
            rows = snap_conn.execute(
                "SELECT message, created_at FROM message_store ORDER BY id"
            ).fetchall()
            snap_conn.close()

            if not rows:
                os.remove(snapshot_path)
                return

            conversation_text = []
            for row in rows:
                msg_data = json.loads(row[0])
                role = msg_data['type']
                content = msg_data['data']['content']
                conversation_text.append(f"{role}: {content}")
            full_text = "\n".join(conversation_text)

            prompt = f"""请根据以下对话内容，生成一段简短的摘要（200字以内），概括主要话题和关键信息：
{full_text}

摘要："""
            response = await self._llm.ainvoke(prompt)
            summary = response.content.strip()

            from memory import get_vector_store
            vs = get_vector_store()
            max_id = await vs.get_max_id(self.session_id) or 0
            new_id = int(max_id) + 1
            await vs.add_summary(self.session_id, str(new_id), summary, metadata={
                "type": "summary",
                "timestamp": int(time.time()),
                "source": "short_term_summary",
            })
            print(f"[记忆] 轮数总结完成: {new_id}")
            os.remove(snapshot_path)
            print(f"[记忆] 已删除快照: {snapshot_path}")
        except Exception as e:
            print(f"[记忆] 轮数总结失败: {e}")
        finally:
            self._summarizing = False

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