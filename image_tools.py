import os
import requests
from langchain.tools import tool
from dashscope import MultiModalConversation
from config import DASHSCOPE_API_KEY


def _get_headers() -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }


def recognize_image(image_url: str) -> str:
    """调用通义千问视觉模型识别图片内容（优先识别身份，其次描述）"""
    if not DASHSCOPE_API_KEY:
        return "图片识别服务未配置"
    try:
        # 先用身份识别 prompt（VL模型自己识别，而非让LLM猜）
        messages = [{
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": "请仔细观察这张图片。如果图中有人物或角色，请直接说出ta的具体身份：叫什么名字？出自哪部作品（动漫/游戏/影视剧）？如果不确定具体身份，请描述外貌特征（发型发色、服饰）和画面内容。80字以内。"}
            ]
        }]
        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages,
            api_key=DASHSCOPE_API_KEY
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            for item in content:
                if item.get("text"):
                    text = item["text"]
                    return text
            return "无法识别图片内容"
        else:
            return f"图片识别失败：{response.message}"
    except Exception as e:
        return f"图片识别出错：{str(e)}"


def _call_vl_identify(image_url: str) -> str:
    """调用通义千问视觉模型识别图中人物/角色身份"""
    if not DASHSCOPE_API_KEY:
        return ""
    try:
        messages = [{
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": "请仔细看这张图片，识别图中的人物或角色。"
                 "这是谁？叫什么名字？出自哪部作品（动漫、游戏、影视剧等）？"
                 "如果无法确定具体身份，请详细描述ta的外貌特征（发型、发色、服饰等）和画面内容。"}
            ]
        }]
        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages,
            api_key=DASHSCOPE_API_KEY
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            for item in content:
                if item.get("text"):
                    return item["text"]
    except Exception as e:
        print(f"[识图] VL身份识别失败: {e}")
    return ""


def _web_search_by_desc(description: str, max_results: int = 3) -> str:
    """用图片描述关键词搜索网页，获取相关信息"""
    if not description or len(description) < 5:
        return ""
    keywords = description.strip()[:60]
    try:
        from urllib.parse import quote_plus
        headers = _get_headers()

        # 用搜狗搜索（对中文内容友好，且可爬取）
        url = f"https://www.sogou.com/web?query={quote_plus(keywords)}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        from bs4 import BeautifulSoup as BS
        soup = BS(resp.text, "html.parser")
        results = []
        for item in soup.select(".vrwrap, .rb, .result")[:max_results]:
            h3 = item.select_one("h3 a, .vr-title a")
            if not h3:
                continue
            title = h3.get_text(strip=True)
            desc_el = item.select_one(".star-wiki, .str-text, .str_info, p")
            desc = desc_el.get_text(strip=True)[:150] if desc_el else ""
            results.append(f"{title}" + (f"\n   {desc}" if desc else ""))

        if results:
            return "相关网页搜索：\n" + "\n".join(results)
    except Exception as e:
        print(f"[识图] 网页搜索失败: {e}")
    return ""


def _bing_visual_search(image_url: str) -> str:
    """Bing 以图搜图"""
    try:
        from urllib.parse import quote
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }
        url = f"https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{quote(image_url)}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        from bs4 import BeautifulSoup as BS
        soup = BS(resp.text, "html.parser")

        lines = []
        title = soup.title.get_text(strip=True) if soup.title else ""
        if title and "必应" not in title:
            lines.append(f"标题: {title}")

        # 提取页面中的图片描述文本
        for tag in soup.select(".caption, .description, .info, .imgcaption"):
            t = tag.get_text(strip=True)
            if t and len(t) > 5:
                lines.append(t[:200])

        # 提取相关页面链接
        count = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            t = a.get_text(strip=True)
            if t and len(t) > 10 and ("http" in href or "www" in href):
                lines.append(f"相关: {t[:100]}")
                count += 1
                if count >= 3:
                    break

        if lines:
            return "Bing以图搜图：\n" + "\n".join(lines)
    except Exception as e:
        print(f"[识图] Bing搜索失败: {e}")
    return ""


@tool
def reverse_image_search(image_url: str) -> str:
    """
    识图搜索：通过图片URL搜索图片来源、识别图中人物/角色身份等。
    当用户想知道"这是谁""这是什么图""这图出自哪里"时使用。
    会综合使用AI视觉识别和网页搜索来获取信息。
    """
    if not image_url:
        return "请提供图片URL"

    parts = []

    # 1. AI视觉识别身份
    print(f"[识图] 开始VL身份识别: {image_url[:60]}...")
    vl_result = _call_vl_identify(image_url)
    if vl_result:
        parts.append(f"AI识别结果：{vl_result}")

    # 2. 用识别结果搜索网页
    if vl_result and len(vl_result) > 10:
        web_result = _web_search_by_desc(vl_result)
        if web_result:
            parts.append(web_result)

    # 3. Bing以图搜图（补充）
    bing_result = _bing_visual_search(image_url)
    if bing_result:
        parts.append(bing_result)

    if not parts:
        return ("未能找到该图片的相关信息。"
                "可以尝试用文字描述图片内容，我来帮你搜索。")

    return "\n\n".join(parts)[:2000]