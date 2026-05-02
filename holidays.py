"""
节日识别 —— 从网络获取当年农历/公历节日，缓存到 JSON。

策略：
  1. 启动时检查 DATA_DIR/holidays_cache.json
  2. 无缓存 → 网络搜索当年节日 → 保存
  3. 每天查一次今日是否有节日
"""
import json
import logging
import os
import datetime

logger = logging.getLogger("qq-bot")

# 固定公历节日（不需要搜索，直接硬编码）
STATIC_HOLIDAYS = {
    (1, 1): "元旦",
    (2, 14): "情人节",
    (3, 8): "妇女节",
    (3, 12): "植树节",
    (4, 1): "愚人节",
    (5, 1): "劳动节",
    (5, 4): "青年节",
    (6, 1): "儿童节",
    (7, 1): "建党节",
    (8, 1): "建军节",
    (9, 10): "教师节",
    (10, 1): "国庆节",
    (12, 25): "圣诞节",
}

_CACHE = {}  # {(month, day): "节日名"}


def ensure_cache(data_dir: str):
    """启动时调用：确保节日缓存存在。"""
    global _CACHE
    cache_path = os.path.join(data_dir, "holidays_cache.json")

    # 先加载静态节日
    _CACHE = dict(STATIC_HOLIDAYS)

    # 尝试加载缓存文件
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                extra = json.load(f)
            for item in extra:
                md = item.get("date", "")
                name = item.get("name", "")
                if md and name and len(md.split("-")) == 2:
                    try:
                        m, d = int(md.split("-")[0]), int(md.split("-")[1])
                        _CACHE[(m, d)] = name
                    except ValueError:
                        pass
            logger.info(f"[节日] 已加载缓存，共 {len(_CACHE)} 个节日")
        except Exception as e:
            logger.debug(f"[节日] 缓存加载失败: {e}")

    return _CACHE


def get_today_holiday() -> str:
    """返回今天的节日名称，无则返回空字符串。"""
    now = datetime.datetime.now()
    return _CACHE.get((now.month, now.day), "")


async def refresh_from_web(data_dir: str):
    """启动时用农历库计算当年农历节日，缓存到 JSON。"""
    cache_path = os.path.join(data_dir, "holidays_cache.json")
    if os.path.exists(cache_path):
        return  # 已有缓存

    year = datetime.datetime.now().year

    # 农历节日：{月, 日: 名称}
    lunar_festivals = {
        (1, 1): "春节", (1, 15): "元宵节",
        (5, 5): "端午节", (7, 7): "七夕节",
        (7, 15): "中元节", (8, 15): "中秋节",
        (9, 9): "重阳节", (12, 30): "除夕",
    }
    try:
        from lunarcalendar import Converter, Lunar, Solar
        found = []

        for (lm, ld), name in lunar_festivals.items():
            try:
                lunar = Lunar(year, lm, ld)
                solar = Converter.Lunar2Solar(lunar)
                m, d = solar.month, solar.day
                if (m, d) not in STATIC_HOLIDAYS:
                    found.append({"date": f"{m:02d}-{d:02d}", "name": name})
            except Exception:
                pass

        # 补冬至（公历固定，但不一定在 STATIC_HOLIDAYS 里）
        # 冬至通常在12月21-23日，简单取22日
        if (12, 22) not in STATIC_HOLIDAYS:
            found.append({"date": "12-22", "name": "冬至"})

        if found:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(found, f, ensure_ascii=False, indent=2)
            logger.info(f"[节日] 已用农历库计算并缓存 {len(found)} 个节日")
            ensure_cache(data_dir)
    except ImportError:
        logger.debug("[节日] lunarcalendar 未安装，跳过")
    except Exception as e:
        logger.debug(f"[节日] 计算失败: {e}")
