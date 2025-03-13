from datetime import datetime

def get_today_str() -> str:
    # yyyymmdd
    # 获取当前的日期和时间
    now = datetime.now()
    # 格式化日期时间为字符串
    formatted_datetime = now.strftime('%Y%m%d')
    return formatted_datetime