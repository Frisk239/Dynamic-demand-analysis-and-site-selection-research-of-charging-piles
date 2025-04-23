import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据
data = {
    'Year': [2016,2017,2018,2019,2020, 2021, 2022, 2023, 2024],
    'EV_Sales': [73, 125, 211, 310, 410, 520, 630, 958, 1316],  # 新能源汽车销量（万辆）
    'Autopilot_Penetration': [5,8,12,18,25,30,34.9,42.4,57.4],  # 自动驾驶功能渗透率（%）
    'Smart_Cabin_Penetration': [10,15,20,25,30,40,59,65,75.6]  # 智能座舱功能渗透率（%）
}

# 创建DataFrame
df = pd.DataFrame(data)

# 可视
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['EV_Sales'], label='新能源汽车销量 (万辆)', marker='o')
plt.plot(df['Year'], df['Autopilot_Penetration'], label='自动驾驶功能渗透率 (%)', marker='s')
plt.plot(df['Year'], df['Smart_Cabin_Penetration'], label='智能座舱功能渗透率 (%)', marker='^')

plt.title('新能源汽车销量与智能化功能渗透率 (2016-2024)')
plt.xlabel('年份')
plt.ylabel('销量/渗透率')
plt.legend()
plt.grid(True)
plt.show()