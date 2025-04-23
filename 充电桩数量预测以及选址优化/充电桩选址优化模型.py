import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager
from pyswarm import pso

# 设置环境变量，避免 KMeans 内存泄漏
os.environ['OMP_NUM_THREADS'] = '1'

# 预定义各省份的经纬度数据
province_coordinates = {
    "北京市": [116.405285, 39.904989],
    "天津市": [117.1994, 39.0851],
    "河北省": [114.4995, 38.1006],
    "山西省": [112.5787, 37.8136],
    "内蒙古自治区": [111.7510, 40.8183],
    "辽宁省": [123.4315, 41.8057],
    "吉林省": [125.3255, 43.8965],
    "黑龙江省": [126.6629, 45.7423],
    "上海市": [121.4737, 31.2304],
    "江苏省": [118.7674, 32.0415],
    "浙江省": [120.1551, 30.2741],
    "安徽省": [117.2830, 31.8612],
    "福建省": [119.2965, 26.0998],
    "江西省": [115.8579, 28.6820],
    "山东省": [117.0204, 36.6683],
    "河南省": [113.7536, 34.7657],
    "湖北省": [114.3423, 30.5459],
    "湖南省": [112.9823, 28.1941],
    "广东省": [113.2665, 23.1322],
    "广西壮族自治区": [108.3200, 22.8240],
    "海南省": [110.3486, 20.0199],
    "重庆市": [106.5505, 29.5630],
    "四川省": [104.0758, 30.6517],
    "贵州省": [106.7074, 26.5982],
    "云南省": [102.7100, 25.0453],
    "西藏自治区": [91.1172, 29.6537],
    "陕西省": [108.9542, 34.2655],
    "甘肃省": [103.8263, 36.0594],
    "青海省": [101.7800, 36.6232],
    "宁夏回族自治区": [106.2309, 38.4872],
    "新疆维吾尔自治区": [87.6168, 43.8256]
}

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取预测结果
def load_predictions():
    # 使用相对路径访问文件
    file_path = os.path.join(current_dir, "预测结果", "最终预测结果.xlsx")
    try:
        predictions_df = pd.read_excel(file_path)
        return predictions_df
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在，请检查路径。")
        exit(1)

# 调整省份名称
def adjust_province_name(name):
    # 补充完整省份名称
    if name == "北京":
        return "北京市"
    elif name == "天津":
        return "天津市"
    elif name == "上海":
        return "上海市"
    elif name == "重庆":
        return "重庆市"
    elif name == "内蒙古":
        return "内蒙古自治区"
    elif name == "广西":
        return "广西壮族自治区"
    elif name == "西藏":
        return "西藏自治区"
    elif name == "宁夏":
        return "宁夏回族自治区"
    elif name == "新疆":
        return "新疆维吾尔自治区"
    elif name == "香港":
        return "香港特别行政区"
    elif name == "澳门":
        return "澳门特别行政区"
    elif name == "台湾":
        return "台湾省"
    else:
        return name + "省"  # 其他省份补充“省”

# 空间聚类（K-Means）
def spatial_clustering(data, n_clusters=5):
    # 提取经纬度数据
    coords = data[['经度', '纬度']].values
    
    # 使用 K-Means 进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(coords)
    
    # 计算每个聚类的中心点
    cluster_centers = kmeans.cluster_centers_
    return data, cluster_centers

# 粒子群优化算法（PSO）
def pso_optimization_per_cluster(cluster_centers, n_stations=1):
    # 定义目标函数
    def evaluate(individual, cluster_center):
        # 计算充电桩位置与当前聚类中心的距离
        distances = np.linalg.norm(cluster_center - np.array(individual))
        return distances

    # 对每个聚类分别进行优化
    best_stations_per_cluster = []
    for cluster_center in cluster_centers:
        # 定义优化问题的边界
        lb = [73, 18]  # 经度和纬度的下限
        ub = [135, 53]  # 经度和纬度的上限

        # 运行 PSO
        best_position, _ = pso(evaluate, lb, ub, args=(cluster_center,))
        best_stations_per_cluster.append(best_position)

    return best_stations_per_cluster

# 绘制中国地图（每个聚类标记一个最优选址）
def plot_china_map_with_optimal_stations(clustered_data, best_stations_per_cluster):
    # 使用相对路径读取中国地图数据
    china_map_path = os.path.join(current_dir, "china.json")
    china_map = gpd.read_file(china_map_path)
    
    # 合并聚类数据到地图数据
    china_map = china_map.merge(clustered_data, left_on='name', right_on='省份', how='left')
    
    # 创建地图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置中文字体
    try:
        # 查找系统中已安装的中文字体
        font_path = font_manager.findfont('SimHei')  # 优先使用 SimHei
    except:
        # 如果找不到 SimHei，使用默认字体
        font_path = font_manager.findfont('sans-serif')
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    
    # 绘制中国地图，根据聚类结果着色
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    legend_handles = []  # 用于存储图例的句柄
    for cluster_id in clustered_data['cluster'].unique():
        cluster_data = china_map[china_map['cluster'] == cluster_id]
        if not cluster_data.empty:  # 确保数据不为空
            cluster_data.plot(ax=ax, color=colors[cluster_id], label=f'Cluster {cluster_id}')
            # 添加图例句柄
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[cluster_id], label=f'Cluster {cluster_id}'))
    
    # 绘制台湾、香港、澳门为灰色
    special_regions = ['台湾省', '香港特别行政区', '澳门特别行政区']
    for region in special_regions:
        region_data = china_map[china_map['name'] == region]
        if not region_data.empty:  # 确保数据不为空
            region_data.plot(ax=ax, color='gray')
    
    # 将 best_stations_per_cluster 转换为二维数组，每行表示一个点的经度和纬度
    best_stations_array = np.array(best_stations_per_cluster).reshape(-1, 2)
    
    # 打印映射后的经纬度，确保其在地图范围内
    print("最优充电桩位置：", best_stations_array)
    
    # 绘制最优充电桩位置
    ax.scatter(best_stations_array[:, 0], best_stations_array[:, 1], color='black', marker='*', s=100, label='最优充电桩')
    # 添加最优充电桩图例句柄
    legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, label='最优充电桩'))

    # 恢复地图的原始比例
    ax.set_aspect('equal')  # 保持地图比例不变形

    # 添加图例
    ax.legend(handles=legend_handles, prop=font_prop)

    # 添加标题
    ax.set_title("中国充电桩选址优化结果（每个聚类一个最优选址）", fontproperties=font_prop)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 加载预测结果
    predictions_df = load_predictions()

    # 打印 predictions_df['省份'] 的内容，检查省份名称
    print("原始省份名称：", predictions_df['省份'].unique())

    # 调整 predictions_df['省份'] 中的名称
    predictions_df['省份'] = predictions_df['省份'].apply(adjust_province_name)
    print("调整后省份名称：", predictions_df['省份'].unique())

    # 添加经纬度数据
    predictions_df['经度'] = predictions_df['省份'].map(lambda x: province_coordinates[x][0])
    predictions_df['纬度'] = predictions_df['省份'].map(lambda x: province_coordinates[x][1])

    # 空间聚类
    clustered_data, cluster_centers = spatial_clustering(predictions_df, n_clusters=5)
    print("聚类中心点：", cluster_centers)

    # 使用 PSO 优化
    best_stations_per_cluster = pso_optimization_per_cluster(cluster_centers, n_stations=1)
    print("每个聚类的最优充电桩位置：", best_stations_per_cluster)

    # 保存结果
    optimized_stations_df = pd.DataFrame({
        '经度': [station[0] for station in best_stations_per_cluster],
        '纬度': [station[1] for station in best_stations_per_cluster]
    })
    output_folder = os.path.join(current_dir, "优化结果")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "充电桩最优位置.xlsx")
    optimized_stations_df.to_excel(output_path, index=False)
    print("优化结果已保存至:", output_path)

    # 绘制中国地图（每个聚类标记一个最优选址）
    plot_china_map_with_optimal_stations(clustered_data, best_stations_per_cluster)
