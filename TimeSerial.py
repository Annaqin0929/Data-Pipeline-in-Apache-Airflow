import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

# 时间序列数据
data = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30])

# 计算不同阶数的 ACF 和 PACF
for p in range(1, 10):
    acf_values = acf(data, nlags=p)
    pacf_values = pacf(data, nlags=p)
    print(f'阶数 {p} 的 ACF: {acf_values}')
    print(f'阶数 {p} 的 PACF: {pacf_values}\n')
