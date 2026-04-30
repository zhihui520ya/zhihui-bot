"""
网页抓取工具包 —— 包含 web_search 和 web_fetch 两个 LangChain 工具。

依赖: requests, beautifulsoup4, lxml (已安装)
"""

import json
import logging
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

logger = logging.getLogger(__name__)

# ========== 配置 ==========

# 请求超时（秒）
REQUEST_TIMEOUT = 15

# 搜索源：当百度 API 不可用时 fallback
SEARCH_FALLBACK_URL = "https://html.duckduckgo.com/html/"

# User-Agent 池（轮换使用，防封）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


def _get_headers() -> dict:
    """生成随机 User-Agent 的请求头"""
    import random
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }


# ==================== 网页搜索 ====================

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    搜索互联网获取信息。当用户问到你不确定的事实、新闻、知识时使用。
    返回标题、链接和摘要列表。

    Args:
        query: 搜索关键词
        max_results: 最多返回几条结果（默认5，最多10）
    """
    if not query or not query.strip():
        return "请输入搜索关键词"

    limit = min(max(1, max_results), 10)
    query = query.strip()

    # Bing 搜索（首选）
    try:
        return _bing_search(query, limit)
    except Exception as e:
        logger.warning(f"Bing搜索失败: {e}")

    # fallback 到 DuckDuckGo
    try:
        return _ddg_search(query, limit)
    except Exception as e:
        return f"搜索失败: {e}"


def _baidu_search(query: str, limit: int) -> str | None:
    """百度搜索（需 BAIDU_API_KEY 和 BAIDU_API_URL 配置）"""
    from config import BAIDU_API_KEY, BAIDU_API_URL
    api_key = BAIDU_API_KEY
    api_url = BAIDU_API_URL
    if not api_key or not api_url:
        return None

    resp = requests.get(
        api_url,
        params={"query": query, "count": limit},
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("result", data.get("data", data.get("items", [])))
    if not results:
        return f"「{query}」没有搜索结果"

    lines = [f"搜索「{query}」结果："]
    for i, item in enumerate(results[:limit]):
        title = item.get("title", item.get("name", ""))
        url = item.get("url", item.get("link", ""))
        snippet = item.get("snippet", item.get("desc", item.get("abstract", "")))
        lines.append(f"{i+1}. {title}")
        if snippet:
            lines.append(f"   {snippet[:120]}")
    return "\n".join(lines)


def _bing_search(query: str, limit: int) -> str:
    """抓取 Bing 搜索结果"""
    url = f"https://www.bing.com/search?q={quote_plus(query)}&count={limit}"
    resp = requests.get(url, headers=_get_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    items = soup.select(".b_algo")
    if not items:
        items = soup.select("[class*='b_algo']")

    lines = [f"搜索「{query}」结果："]
    count = 0
    for item in items:
        if count >= limit:
            break
        h2 = item.select_one("h2")
        if not h2:
            continue
        a = h2.select_one("a")
        if not a:
            continue
        title = a.get_text(strip=True)
        href = a.get("href", "")
        snippet_el = item.select_one(".b_caption p, .b_lineclamp2")
        snippet = snippet_el.get_text(strip=True)[:120] if snippet_el else ""
        if title:
            lines.append(f"{count+1}. {title}")
            if snippet:
                lines.append(f"   {snippet}")
            count += 1

    if count == 0:
        return f"「{query}」没有搜索结果"
    return "\n".join(lines)


def _ddg_search(query: str, limit: int) -> str:
    """DuckDuckGo HTML 搜索（备用）"""
    resp = requests.post(
        SEARCH_FALLBACK_URL,
        data={"q": query},
        headers=_get_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results = soup.select(".result")
    if not results:
        results = soup.select("[class*='result']")

    lines = [f"搜索「{query}」结果："]
    count = 0
    for r in results:
        if count >= limit:
            break
        title_el = r.select_one("h2 a, .result__title a, a")
        snippet_el = r.select_one(".result__snippet, .snippet, p")

        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        url = title_el.get("href", "")
        # DuckDuckGo 的链接经过重定向，需要提取真实 URL
        if "uddg=" in url:
            from urllib.parse import parse_qs, urlparse as up
            parsed = up(url)
            qs = parse_qs(parsed.query)
            url = qs.get("uddg", [url])[0]
        snippet = snippet_el.get_text(strip=True)[:120] if snippet_el else ""

        if title:
            lines.append(f"{count+1}. {title}")
            if snippet:
                lines.append(f"   {snippet}")
            count += 1

    if count == 0:
        return f"「{query}」没有搜索结果"

    return "\n".join(lines)


# ==================== 网页内容抓取 ====================

@tool
def web_fetch(url: str, max_chars: int = 3000) -> str:
    """
    抓取指定网页的内容并返回纯文本。当需要了解某篇文章、新闻、页面的详细内容时使用。
    通常先调 web_search 找到链接，再用 web_fetch 读取内容。

    Args:
        url: 要抓取的网页完整 URL（以 http:// 或 https:// 开头）
        max_chars: 最多返回多少字符（默认3000，最大10000）
    """
    if not url or not url.strip():
        return "请输入要抓取的 URL"

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    limit = min(max(100, max_chars), 10000)

    try:
        resp = requests.get(url, headers=_get_headers(), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        # 检测编码
        if resp.encoding and resp.encoding.lower() != "utf-8":
            resp.encoding = resp.encoding or "utf-8"

        content_type = resp.headers.get("Content-Type", "")
        text = resp.text

        # 尝试按编码解码
        try:
            text = resp.content.decode("utf-8")
        except Exception:
            try:
                text = resp.content.decode("gbk")
            except Exception:
                text = resp.text

        soup = BeautifulSoup(text, "lxml")

        # 移除无用元素
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "noscript", "iframe", "svg", "form", "button", "input"]):
            tag.decompose()

        # 提取标题
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)

        # 提取正文（优先 article / main 区域）
        body = soup.find("article") or soup.find("main") or soup.find("body") or soup
        content = body.get_text(separator="\n", strip=True)

        # 清理多余空行
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        content = "\n".join(lines)

        # 限制长度
        if len(content) > limit:
            content = content[:limit] + "\n\n...（内容过长已截断）"

        result = []
        if title:
            result.append(f"标题: {title}")
        result.append(f"来源: {url}")
        result.append("")
        result.append(content)

        return "\n".join(result)

    except requests.exceptions.Timeout:
        return f"抓取 {url} 超时（{REQUEST_TIMEOUT}秒）"
    except requests.exceptions.ConnectionError:
        return f"无法连接 {url}"
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        return f"抓取 {url} 失败（HTTP {status}）"
    except Exception as e:
        return f"抓取 {url} 时出错: {e}"


# ==================== 工具列表 ====================

WEB_TOOLS = [web_search, web_fetch]
