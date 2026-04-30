"""
中央配置 —— 所有可调参数统一入口。
简单值通过 .env 可覆盖，复杂数据结构直接定义在此。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== 情绪参数 ====================

# 半衰期（秒）
EMOTION_HALF_LIFE = {
    "生气": 1800,
    "醋意": 2400,
    "委屈": 7200,
    "害羞": 5400,
    "开心": 3600,
    "撒娇": 3600,
}

# 基底情绪（长时间不聊天也不低于此值）
EMOTION_BASELINE = {
    "醋意": 0.8,
    "撒娇": 0.5,
    "害羞": 0.3,
    "开心": 0.2,
    "生气": 0.0,
    "委屈": 0.0,
}

RAPID_DECAY_FACTOR = float(os.getenv("RAPID_DECAY_FACTOR", "0.9"))
RAPID_DECAY_TIMEOUT = int(os.getenv("RAPID_DECAY_TIMEOUT", "1800"))
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.4"))
DELTA_MAX = float(os.getenv("DELTA_MAX", "5"))
FORGETTING_PROBABILITY = float(os.getenv("FORGETTING_PROBABILITY", "0.2"))
ANCHOR_RESURRECT_FACTOR = float(os.getenv("ANCHOR_RESURRECT_FACTOR", "0.4"))

# ==================== 好感度参数 ====================
AFFECTION_HALF_LIFE = int(os.getenv("AFFECTION_HALF_LIFE", "604800"))
AFFECTION_MIN = int(os.getenv("AFFECTION_MIN", "0"))
AFFECTION_MAX = int(os.getenv("AFFECTION_MAX", "10000"))

# ==================== 回复延迟（秒） ====================
MAIN_DELAY = float(os.getenv("MAIN_DELAY", "6.0"))
SILENT_DELAY = float(os.getenv("SILENT_DELAY", "1.0"))
MAX_REPLY_MESSAGES = int(os.getenv("MAX_REPLY_MESSAGES", "3"))
SEND_DELAY_MIN = float(os.getenv("SEND_DELAY_MIN", "0.8"))
SEND_DELAY_MAX = float(os.getenv("SEND_DELAY_MAX", "3.8"))

# ==================== 回复概率 ====================
PRIVATE_REPLY_PROBABILITY = float(os.getenv("PRIVATE_REPLY_PROBABILITY", "1.0"))
GROUP_REPLY_PROBABILITY = float(os.getenv("GROUP_REPLY_PROBABILITY", "0.1"))

# ==================== 记忆参数 ====================
TOPIC_SWITCH_TIME_THRESHOLD = int(os.getenv("TOPIC_SWITCH_TIME_THRESHOLD", "90"))
TOPIC_SWITCH_SIMILARITY = float(os.getenv("TOPIC_SWITCH_SIMILARITY", "0.2"))
INCREMENTAL_ORGANIZE_INTERVAL = int(os.getenv("INCREMENTAL_ORGANIZE_INTERVAL", "10800"))

# ==================== 路径 ====================
DATA_DIR = os.getenv("DATA_DIR", ".")

# ==================== 图片识别 ====================
MAX_IMAGE_RECOGNITION = int(os.getenv("MAX_IMAGE_RECOGNITION", "3"))

# ==================== API 密钥 ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY", "")
BAIDU_API_URL = os.getenv("BAIDU_API_URL", "")
HEWEATHER_KEY = os.getenv("HEWEATHER_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# ==================== NapCat ====================
NAPCAT_TOKEN = os.getenv("NAPCAT_TOKEN", "")
NAPCAT_BASE_URL = os.getenv("NAPCAT_BASE_URL", "http://127.0.0.1:3000")

# ==================== QQ 空间 ====================
# 留空则启动时自动从 NapCat 获取
QZONE_COOKIES = os.getenv("QZONE_COOKIES", "")

# ==================== 社交过滤器 + 认知资源 ====================
# 认知资源(CAW)容量范围：好感度 0~10000 映射到此范围
COGNITIVE_CAPACITY_MIN = float(os.getenv("COGNITIVE_CAPACITY_MIN", "20.0"))
COGNITIVE_CAPACITY_MAX = float(os.getenv("COGNITIVE_CAPACITY_MAX", "100.0"))
# 认知资源恢复速率（单位/秒）
COGNITIVE_RECOVERY_RATE = float(os.getenv("COGNITIVE_RECOVERY_RATE", "0.016667"))
# 爆发阈值（资源比例，0~1）
COGNITIVE_BURST_THRESHOLD = float(os.getenv("COGNITIVE_BURST_THRESHOLD", "0.2"))
COGNITIVE_BURST_EXIT_THRESHOLD = float(os.getenv("COGNITIVE_BURST_EXIT_THRESHOLD", "0.5"))

# ==================== 机器人基础配置 ====================
# 机器人自己的QQ号（用于检测是否被@）
BOT_QQ = int(os.getenv("BOT_QQ", "0"))

# 自定义触发符号：消息以这些符号开头时，强制回复（群聊中生效）
TRIGGER_SYMBOLS = {"/", "#", "."}

# 管理员 QQ 号列表（只有列表内的用户可以使用 /admin 调试指令）
ADMIN_QQ = set(int(x.strip()) for x in os.getenv("ADMIN_QQ", "").split(",") if x.strip())

# ==================== 安全限制 ====================
# 请求体大小限制（防止大 POST 攻击）
MAX_BODY_SIZE = int(os.getenv("MAX_BODY_SIZE", "100000"))
# 令牌桶限流
RATE_LIMIT_RATE = float(os.getenv("RATE_LIMIT_RATE", "2.0"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "5"))
# 输入字段校验
MAX_RAW_MESSAGE_LENGTH = int(os.getenv("MAX_RAW_MESSAGE_LENGTH", "2000"))
MAX_MESSAGE_SEGMENTS = int(os.getenv("MAX_MESSAGE_SEGMENTS", "50"))

# ==================== 权限控制 ====================
# 私聊白名单：允许回复的QQ号列表（空集合=允许所有）
# 可通过 .env 设置：PRIVATE_WHITELIST=123456,789012
PRIVATE_WHITELIST = set(int(x.strip()) for x in os.getenv("PRIVATE_WHITELIST", "").split(",") if x.strip())

# 群聊白名单：允许回复的群号列表（空集合=允许所有）
GROUP_WHITELIST = set(int(x.strip()) for x in os.getenv("GROUP_WHITELIST", "").split(",") if x.strip())

# 群内用户白名单：限制只有特定用户在群内才能触发回复
# 格式：{"群号": {QQ号, QQ号}}
GROUP_USER_WHITELIST = {}

# 私聊黑名单
PRIVATE_BLACKLIST = set(int(x.strip()) for x in os.getenv("PRIVATE_BLACKLIST", "").split(",") if x.strip())

# 群聊黑名单
GROUP_BLACKLIST = set(int(x.strip()) for x in os.getenv("GROUP_BLACKLIST", "").split(",") if x.strip())

# 群内用户黑名单
GROUP_USER_BLACKLIST = {}

# ==================== 邮件 (SMTP) ====================
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "QQ Bot")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "false").lower() in ("true", "1", "yes")
SMTP_USE_STARTTLS = os.getenv("SMTP_USE_STARTTLS", "true").lower() in ("true", "1", "yes")

# ==================== IMAP ====================
IMAP_HOST = os.getenv("IMAP_HOST", "imap.qq.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
