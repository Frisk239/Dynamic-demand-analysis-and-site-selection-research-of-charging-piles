# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# 设置兼容中文和负号的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = True  # 确保显示负号

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 定义各省份面积
area_dict = {
    '北京市': 16410.54,
    '天津市': 11966.45,
    '河北省': 187700.00,
    '山西省': 156000.00,
    '内蒙古自治区': 1183000.00,
    '辽宁省': 148000.00,
    '吉林省': 187400.00,
    '黑龙江省': 454800.00,
    '上海市': 6340.50,
    '江苏省': 102600.00,
    '浙江省': 101800.00,
    '安徽省': 139600.00,
    '福建省': 124000.00,
    '江西省': 166900.00,
    '山东省': 157100.00,
    '河南省': 167000.00,
    '湖北省': 185900.00,
    '湖南省': 211800.00,
    '广东省': 179800.00,
    '广西壮族自治区': 236000.00,
    '海南省': 35400.00,
    '重庆市': 82400.00,
    '四川省': 486000.00,
    '贵州省': 176100.00,
    '云南省': 394100.00,
    '西藏自治区': 1228400.00,
    '陕西省': 205600.00,
    '甘肃省': 454000.00,
    '青海省': 721200.00,
    '宁夏回族自治区': 66400.00,
    '新疆维吾尔自治区': 1664900.00,
    '台湾省': 36193.00,
    '香港': 1106.34,
    '澳门': 32.90
}

# 加载地理数据 - 使用相对路径
print("正在加载地理数据...")
china = gpd.read_file(os.path.join(script_dir, 'china.json'))
china = china[~china['name'].isna() & (china['name'] != '')].copy().reset_index(drop=True)

# 加载充电桩数据 - 使用相对路径
print("\n正在加载充电桩数据...")
# 历史数据文件（同一目录下）
hist_data_path = os.path.join(script_dir, "16-22年各省份公共充电桩保有量.xlsx")
# 预测数据文件（在预测结果子目录下）
pred_data_path = os.path.join(script_dir, "预测结果", "最终预测结果.xlsx")

hist_data = pd.read_excel(hist_data_path)
pred_data = pd.read_excel(pred_data_path)

# 数据预处理
def preprocess_data(df):
    df = df.copy()
    df.columns = ['省份'] + [str(col) if isinstance(col, int) else col for col in df.columns[1:]]
    df['省份'] = df['省份'].str.strip()
    return df

hist_data = preprocess_data(hist_data)
pred_data = preprocess_data(pred_data)
all_data = pd.merge(hist_data, pred_data, on='省份', how='outer', suffixes=('', '_pred'))

# 省份转换器
province_converter = {
    '北京': '北京市', '天津': '天津市', '上海': '上海市', '重庆': '重庆市',
    '内蒙古': '内蒙古自治区', '广西': '广西壮族自治区', '西藏': '西藏自治区',
    '宁夏': '宁夏回族自治区', '新疆': '新疆维吾尔自治区', '台湾': '台湾省',
    '香港': '香港', '澳门': '澳门'
}

def convert_province_name(name):
    name = name.strip()
    return province_converter.get(name, f"{name}省" if not name.endswith(('省', '市', '自治区')) else name)

# 数据计算函数，确保顺序一致
def calculate_density(data, year):
    densities = pd.Series(index=china['name'], dtype=float).fillna(0)
    for _, row in data.iterrows():
        province_name = convert_province_name(row['省份'])
        if province_name in densities.index:
            try:
                count = row.get(str(year), row.get(f"{year}_pred", 0))
                densities[province_name] = float(count) / area_dict.get(province_name, 1)
            except Exception as e:
                print(f"计算{province_name} {year}年密度时出错: {e}")
    return densities.reindex(china['name']).fillna(0).values

# 创建颜色映射
def create_optimized_colormap():
    colors = ["#f7f7f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f", "#3f007d"]
    cmap = LinearSegmentedColormap.from_list("custom_purple", colors)
    cmap.set_bad(color='#808080', alpha=0.7)
    return cmap

# 初始化绘图
fig, ax = plt.subplots(figsize=(14, 12), dpi=120)
cmap = create_optimized_colormap()
norm = LogNorm(vmin=0.001, vmax=50)
initial_year = 2016
densities = calculate_density(all_data, initial_year)

# 隐藏坐标轴刻度
ax.set_xticks([])  # 隐藏x轴刻度
ax.set_yticks([])  # 隐藏y轴刻度
ax.set_frame_on(False)  # 隐藏边框

# 初始绘图并保存返回的plot对象
masked_densities = np.ma.masked_where(np.isnan(densities) | np.isin(china['name'], ['台湾省', '香港', '澳门']), densities)
plot = china.plot(column=masked_densities, ax=ax, legend=True, cmap=cmap, norm=norm)

# 获取色彩条对象并调整位置
cbar = plot.get_figure().get_axes()[1]  # 获取第二个轴，即色彩条
cbar_pos = cbar.get_position()
cbar.set_position([cbar_pos.x0, cbar_pos.y0 + 0.05, cbar_pos.width, cbar_pos.height * 0.9])  # 上移色彩条

# 在色彩条上方添加单位 - 使用text方法添加
fig.text(cbar_pos.x0 + cbar_pos.width/2, cbar_pos.y0 + cbar_pos.height + 0.02, 
         '单位：个/平方公里', 
         ha='center', va='bottom', fontsize=12)

# 滑动条
ax_slider = plt.axes([0.25, 0.1, 0.6, 0.03])
year_slider = Slider(ax=ax_slider, label='选择年份', valmin=2016, valmax=2030, valinit=initial_year, valstep=1)

def update(val):
    year = int(year_slider.val)
    new_densities = calculate_density(all_data, year)
    
    # 更新数据而不重新创建色彩条
    new_masked = np.ma.masked_where(
        np.isnan(new_densities) | np.isin(china['name'], ['台湾省', '香港', '澳门']),
        new_densities
    )
    
    # 清除当前绘图
    for coll in ax.collections:
        coll.remove()
    
    # 重新绘制地图，但不创建新的色彩条
    china.plot(column=new_masked, ax=ax, legend=False, cmap=cmap, norm=norm)
    
    # 更新标题，2023年及以后添加"(预测)"
    title = f"{year}年中国各省份充电桩密度分布"
    if year >= 2023:
        title += "（预测）"
    ax.set_title(title, fontsize=16)
    
    # 更新色彩条范围（如果需要）
    cbar.set_clim(vmin=0.001, vmax=50)
    
    fig.canvas.draw_idle()

# 绑定事件
year_slider.on_changed(update)
plt.show()