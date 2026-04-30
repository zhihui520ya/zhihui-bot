# tools.py
import os
import random
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from config import BAIDU_API_KEY, BAIDU_API_URL, HEWEATHER_KEY

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气（使用和风天气 API）"""
    if not HEWEATHER_KEY:
        return "天气服务未配置，请在 .env 文件中设置 HEWEATHER_KEY"

    print(f"[工具调用] 正在查询 {city} 的天气...")
    try:
        # 第一步：获取城市ID
        geo_url = "https://geoapi.qweather.com/v2/city/lookup"
        geo_params = {
            "location": city,
            "key": HEWEATHER_KEY
        }
        geo_resp = requests.get(geo_url, params=geo_params, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if geo_data.get("code") != "200" or not geo_data.get("location"):
            return f"未找到城市“{city}”，请尝试使用更完整的名称（如“北京市”）。"
        location_id = geo_data["location"][0]["id"]
        city_name = geo_data["location"][0]["name"]

        # 第二步：获取实时天气
        weather_url = "https://devapi.qweather.com/v7/weather/now"
        weather_params = {
            "location": location_id,
            "key": HEWEATHER_KEY
        }
        weather_resp = requests.get(weather_url, params=weather_params, timeout=10)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        if weather_data.get("code") != "200":
            return f"获取天气失败：{weather_data.get('code')}"

        now = weather_data.get("now", {})
        temp = now.get("temp")
        text = now.get("text")
        feels_like = now.get("feelsLike")
        wind_dir = now.get("windDir")
        wind_scale = now.get("windScale")
        humidity = now.get("humidity")

        result = (
            f"{city_name} 当前天气：{text}\n"
            f"温度：{temp}℃（体感 {feels_like}℃）\n"
            f"风向：{wind_dir}，风力：{wind_scale}级\n"
            f"湿度：{humidity}%"
        )
        print(f"[工具结果] {result}")
        return result
    except Exception as e:
        error_msg = f"天气查询失败：{str(e)}"
        print(f"[工具错误] {error_msg}")
        return error_msg

def _baidu_search_api(query: str) -> str:
    """内部版：使用百度智能云 AI 搜索 API 获取实时信息"""
    if not BAIDU_API_KEY or not BAIDU_API_URL:
        return ""

    print(f"[工具-新闻] 正在调用百度 AI 搜索 API：{query}")
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BAIDU_API_KEY}"
        }
        payload = {
            "messages": [
                {"role": "user", "content": query}
            ],
            "stream": False
        }
        resp = requests.post(BAIDU_API_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result_text = ""
        if "choices" in data and len(data["choices"]) > 0:
            result_text = data["choices"][0].get("message", {}).get("content", "")
        elif "result" in data:
            result_text = data["result"]
        elif "data" in data:
            result_text = data["data"]
        else:
            result_text = str(data)

        return result_text.strip()
    except Exception as e:
        print(f"[工具-新闻] 百度 AI 搜索 API 失败：{e}")
        return ""


def _baidu_search_fallback(query: str) -> str:
    """内部版：备用爬虫搜索"""
    print(f"[工具-新闻] 正在使用爬虫搜索：{query}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = f"https://www.baidu.com/s?wd={requests.utils.quote(query)}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')

        results = []
        for item in soup.select('.result, .c-container')[:3]:
            title_el = item.select_one('h3 a')
            title_text = title_el.get_text(strip=True) if title_el else '无标题'
            abstract_el = item.select_one('.c-abstract')
            abstract_text = abstract_el.get_text(strip=True) if abstract_el else ''
            results.append(f"{title_text}\n{abstract_text}")

        if not results:
            return ""
        return "\n---\n".join(results)
    except Exception as e:
        print(f"[工具-新闻] 爬虫搜索失败：{e}")
        return ""


@tool
def get_news(category: str = "科技") -> str:
    """搜索新闻或实时信息。自动使用 Bing 搜索抓取结果。"""
    print(f"[工具调用] get_news(category={category})")

    from web_tools import web_search
    return web_search.invoke({"query": category + " 新闻", "max_results": 5})

