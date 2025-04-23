import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib

# ================== 全局字体设置 ==================
# 设置中文字体（需确保系统已安装）
plt.rcParams['font.family'] = 'SimHei'  # 直接指定字体名称
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

# ================== 数据准备 ==================
data = {
    "省份": ["北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", 
           "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", 
           "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州", 
           "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"],
    "2016": [21940,6782,7307,3349,92,2520,47,82,16444,15869,5246,6756,2810,446,12340,2132,4266,2041,21108,393,365,2362,2986,282,412,9,1973,339,248,189,119],
    "2017": [30363,9788,9875,5244,177,3184,259,579,26314,22075,9866,9909,5046,1357,17557,3702,6214,3655,29262,951,916,4949,4731,829,1429,9,3774,1226,340,199,124],
    "2018": [41644,11209,11957,6500,1224,4280,733,1883,39303,30333,14226,10228,7942,2943,20798,8131,9722,5184,35928,1771,1403,8538,8169,1779,1987,17,8705,2327,445,233,207],
    "2019": [59060,16687,22307,12027,2525,6044,1763,2666,55113,60509,29138,25754,17074,6744,32130,15968,17592,10498,62834,3596,3354,11245,14150,3207,3702,18,14857,3426,947,363,913],
    "2020": [87634,27846,31804,19455,3490,8258,3768,4860,85538,77053,61542,38959,25926,11343,48890,32816,33408,18554,85874,7580,5813,17533,23872,4902,6004,193,25569,4377,1115,1510,1897],
    "2021": [96840,33383,38491,25474,4500,10285,3811,7070,103249,97265,82041,58307,39853,15547,60251,43556,58627,27095,181846,14026,14042,20608,38312,9355,13600,391,34738,5637,1784,2190,3206],
    "2022": [110145,46565,48950,33593,7696,12907,6810,9015,122235,129677,125918,84129,67299,26739,89965,68016,101163,41336,382960,33246,27596,38512,61416,20261,32146,557,48277,8088,2326,3329,4802]
}

df = pd.DataFrame(data)

# ================== 数据处理 ==================
# 提取西藏数据（2017年设为缺失）
tibet_data = df[df['省份'] == '西藏'].iloc[:, 1:].values.flatten()
years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022])
valid_years = years[years != 2017]
valid_values = tibet_data[years != 2017]

# ================== 插值计算 ==================
spline = make_interp_spline(valid_years, valid_values, k=1)
interpolated_value = spline(2017).item()
interpolated_value = max(9, min(interpolated_value, 17))  # 应用约束

# 更新数据
df['2017'] = df['2017'].astype(float)
df.loc[df['省份'] == '西藏', '2017'] = round(interpolated_value, 2)

# ================== 可视化呈现 ==================
plt.figure(figsize=(12, 7), dpi=100)

# 绘制主曲线
plt.plot(valid_years, valid_values, 'o--', color='#2c7bb6', 
         markersize=10, linewidth=2.5, label='原始趋势线')

# 突出显示插值点
plt.scatter(2017, interpolated_value, s=180, color='#d7191c', 
           marker='*', edgecolor='black', zorder=5, 
           label=f'2017年插值 ({interpolated_value:.1f})')

# 标注约束范围
plt.fill_between([2016, 2022], 9, 17, color='gray', 
                alpha=0.1, label='允许范围')

# 辅助线
plt.axhline(y=9, color='#1a9641', linestyle=':', linewidth=1.5)
plt.axhline(y=17, color='#fdae61', linestyle=':', linewidth=1.5)

# 图表装饰
plt.title('西藏充电桩保有量插值补全（2016-2022）', fontsize=16, pad=20)
plt.xlabel('年份', fontsize=12, labelpad=10)
plt.ylabel('充电桩数量（个）', fontsize=12, labelpad=10)
plt.xticks(years, [f'{y}年' for y in years], rotation=45)

# 图例优化
legend = plt.legend(fontsize=10, frameon=True, 
                   shadow=True, borderpad=1)
legend.get_frame().set_facecolor('#f7f7f7')

# 网格和边距
plt.grid(True, linestyle='--', alpha=0.6)
plt.margins(x=0.05, y=0.1)

# 保存输出
plt.tight_layout()
plt.savefig('Tibet_Charging_Pile_Interpolation.png', 
           dpi=300, bbox_inches='tight', transparent=False)
plt.show()