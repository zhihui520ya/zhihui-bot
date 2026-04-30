"""
共享可变状态 —— 供 reply_engine 和 main 跨模块访问。

提取自 main.py 的全局缓存 + 调度锁 + 安全任务包装，避免循环依赖。
"""
import asyncio
import traceback

# ========== 调度锁（保护 cached_data、pending_tasks 等全局缓存的竞态） ==========
_schedule_locks: dict[str, asyncio.Lock] = {}
_schedule_locks_guard = asyncio.Lock()


async def _get_schedule_lock(session_id: str) -> asyncio.Lock:
    async with _schedule_locks_guard:
        if session_id not in _schedule_locks:
            _schedule_locks[session_id] = asyncio.Lock()
        return _schedule_locks[session_id]


# ========== 安全的后台任务包装（捕获并记录所有非 CancelledError 异常） ==========

def safe_create_task(coro, name=None):
    """
    创建后台任务，自动捕获并记录所有意外异常。
    相比裸 asyncio.create_task，防止任务静默消亡而不被感知。
    """
    async def _wrapper():
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            task_name = name or getattr(coro, '__name__', '?')
            print(f"[后台任务] 未捕获异常 ({task_name}):")
            traceback.print_exc()
    return asyncio.create_task(_wrapper(), name=name)


# ========== 回复调度全局缓存 ==========

# session_id -> asyncio.Task (主延迟任务)
pending_tasks: dict[str, asyncio.Task] = {}
# session_id -> tuple (缓存的回复数据)
cached_data: dict[str, tuple] = {}
# 记录正在发送回复的会话
sending_sessions: set[str] = set()
# session_id -> asyncio.Task (静默等待任务)
silent_tasks: dict[str, asyncio.Task] = {}
# session_id -> 连续输入中通知计数
input_status_count: dict[str, int] = {}

# ========== 回复执行锁（防止同一会话并发 send_reply） ==========
_reply_locks: dict[str, asyncio.Lock] = {}
_reply_locks_guard = asyncio.Lock()


async def _get_reply_lock(session_id: str) -> asyncio.Lock:
    async with _reply_locks_guard:
        if session_id not in _reply_locks:
            _reply_locks[session_id] = asyncio.Lock()
        return _reply_locks[session_id]


# ========== 情感锚点复活冷却 ==========
ANCHOR_RESURRECT_COOLDOWN = 3600  # 1 小时
# session_id -> {anchor_id: timestamp}
_recently_resurrected: dict[str, dict[str, float]] = {}
