import numpy as np
import pandas as pd
import tushare as ts


# 初始化pro接口
pro = ts.pro_api('02f2137ca44c9bef54b97948edee6fd951e1b1d9ff9ea6e5e9e7d97c')

# 拉取数据
df = pro.stock_basic(**{
    "ts_code": "688981.SH",
    "name": "",
    "exchange": "",
    "market": "",
    "is_hs": "",
    "list_status": "",
    "limit": "",
    "offset": ""
}, fields=[
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "list_date"
])
print(df)

