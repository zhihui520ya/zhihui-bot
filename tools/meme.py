"""
本地表情包工具 —— 让知慧能从本地文件夹发送表情包图片。

目录结构：
  memes/
  ├── index.json          # 分类名 → 描述文本
  ├── wuliao/             # 每类一个文件夹，放若干图片
  ├── happy/
  └── ...

两个 @tool：
  - send_meme(category)   → 发一张该分类的随机图片
  - search_meme(desc)     → 根据描述找最匹配的分类
"""
import json
import logging
import os
import random
from pathlib import Path

from langchain.tools import tool

logger = logging.getLogger("qq-bot")

MEMES_DIR = Path(__file__).resolve().parent.parent / "memes"
INDEX_PATH = MEMES_DIR / "index.json"

# 支持的图片扩展名
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# ========== 启动时确保目录和索引存在 ==========
MEMES_DIR.mkdir(parents=True, exist_ok=True)
if not INDEX_PATH.exists():
    INITIAL_INDEX = {
        "wuliao": "很无聊、想躺平的时候",
        "taiqiangle": "对方太强了，表达佩服",
        "happy": "开心、庆祝、成功的时候",
        "sad": "伤心、遗憾、需要安慰的时候",
        "angry": "生气、不满、想吐槽的时候",
        "shy": "害羞、被夸、不好意思的时候",
        "sleep": "困了、熬夜、想睡觉的时候",
        "meow": "卖萌、撒娇、装可爱的时候",
    }
    INDEX_PATH.write_text(json.dumps(INITIAL_INDEX, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[MemeTools] 已创建默认索引: {INDEX_PATH}")
    # 创建示例目录
    for cat in INITIAL_INDEX:
        cat_dir = MEMES_DIR / cat
        cat_dir.mkdir(exist_ok=True)


def _get_image_files(category: str) -> list[Path]:
    """获取某分类下的所有图片文件列表。"""
    cat_dir = MEMES_DIR / category
    if not cat_dir.is_dir():
        return []
    files = []
    for f in cat_dir.iterdir():
        if f.suffix.lower() in _IMAGE_EXTS and f.is_file():
            files.append(f)
    return files


def _load_index() -> dict[str, str]:
    """加载分类索引。"""
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@tool
def send_meme(category: str) -> str:
    """根据分类名发送一张本地表情包图片。

    调用前应先用 search_meme 找到合适的分类名。
    支持的分类可通过 search_meme 查询。

    Args:
        category: 分类目录名（如 wuliao、happy、taiqiangle）
    """
    index = _load_index()
    if category not in index:
        available = "、".join(index.keys()) if index else "（暂无分类）"
        return f"没有「{category}」这个分类。可用分类：{available}"

    files = _get_image_files(category)
    if not files:
        return f"分类「{category}」下还没有表情包，往 memes/{category}/ 里丢几张图吧"

    chosen = random.choice(files)
    cq_code = f"[CQ:image,file=file://{chosen.resolve()}]"
    logger.debug(f"[MemeTools] send_meme({category}) → {chosen.name}")
    return cq_code


@tool
def search_meme(description: str) -> str:
    """根据文字描述搜索最匹配的表情包分类名。

    描述使用场景或情绪，例如「很无聊」「对方太强了」「好开心」。
    返回最匹配的分类名和描述，调用方再用 send_meme 发送。

    Args:
        description: 场景或情绪描述
    """
    index = _load_index()
    if not index:
        return "还没有表情包分类，往 memes/ 里建文件夹丢图吧"

    # 按汉字/单词拆分为关键词，对分类名和描述做评分
    chars = list(description.strip())  # 每个字符作为关键词
    bi_grams = {description[i:i+2] for i in range(len(description)-1)}  # 二元组加分
    best_cat = None
    best_score = 0

    for cat, desc in index.items():
        score = 0
        text = cat + desc  # 合并搜索
        for c in chars:
            if c in text:
                score += 1
        for bg in bi_grams:
            if bg in text:
                score += 3
        if score > best_score:
            best_score = score
            best_cat = cat

    if best_cat and best_score > 0:
        return f"{best_cat}：{index[best_cat]}"
    # 无关键词匹配，返回所有分类让 LLM 选择
    all_cats = "\n".join(f"  {k}：{v}" for k, v in index.items())
    return f"未找到直接匹配的分类，可用分类如下：\n{all_cats}"
