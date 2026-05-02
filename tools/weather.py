# tools/weather.py
import os
import requests
from langchain.tools import tool

HEWEATHER_KEY = os.getenv("HEWEATHER_KEY", "")   # 和风天气 API Key

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

@tool
def get_news(category: str = "科技") -> str:
    """搜索新闻或实时信息。自动使用 Bing 搜索抓取结果。"""
    print(f"[工具调用] get_news(category={category})")

    from web_tools import web_search
    return web_search.invoke({"query": category + " 新闻", "max_results": 5})

