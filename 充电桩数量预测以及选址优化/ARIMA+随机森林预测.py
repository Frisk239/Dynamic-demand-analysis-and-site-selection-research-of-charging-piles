# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from scipy.ndimage import gaussian_filter1d
import libpysal.weights as weights
import os
import warnings
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Any

warnings.filterwarnings("ignore")

# 配置参数
PROVINCE_POLICY = {
    '西藏': 1.5, '青海': 1.2, '宁夏': 1.1,
    # 其他省份默认为1.0
}

PROVINCE_CAPACITY = {
    '北京': 500000, '上海': 600000, '广东': 1000000,
    '西藏': 100000, '青海': 150000, '宁夏': 200000,
    # 其他省份默认容量
}

def prepare_spatial_weights(provinces: List[str], adjacency: Dict[str, List[str]]) -> Any:
    """准备空间权重矩阵 - 修正版"""
    try:
        # 创建邻接字典
        adj_dict = {}
        for province in provinces:
            neighbors = adjacency.get(province, [])
            # 只保留存在于provinces列表中的邻接省份
            valid_neighbors = [n for n in neighbors if n in provinces]
            adj_dict[province] = valid_neighbors
        
        # 使用Queen邻接创建权重矩阵
        w = weights.Queen.from_adjacency(adj_dict)
        return w
    except Exception as e:
        print(f"创建空间权重矩阵失败: {str(e)}")
        # 返回一个空权重矩阵作为后备
        return weights.W({})

def spatial_adjustment(province: str, 
                      value: float,
                      w: Any,
                      province_data: Dict[str, float]) -> float:
    """
    空间调整函数 - 简化版
    参数:
        province: 当前省份
        value: 待调整的值
        w: 空间权重矩阵
        province_data: 各省份最新数据
    返回:
        调整后的值
    """
    try:
        if province not in w.neighbors or len(province_data) < 3:
            return value
        
        # 获取邻域省份的值
        neighbor_values = []
        for neighbor in w.neighbors[province]:
            if neighbor in province_data:
                neighbor_values.append(province_data[neighbor])
        
        if not neighbor_values:
            return value
            
        neighbor_avg = np.mean(neighbor_values)
        
        # 简单调整: 取当前值和邻域值的加权平均
        return 0.8 * value + 0.2 * neighbor_avg
    
    except Exception as e:
        print(f"{province} 空间调整失败: {str(e)}")
        return value

def preprocess_data(df: pd.DataFrame, province: str) -> pd.Series:
    """数据预处理函数"""
    try:
        # 确保完整的时间范围
        full_years = pd.DataFrame({'年份': [str(y) for y in range(2016, 2023)]})
        province_data = df[df['省份'] == province].copy()
        province_data['年份'] = province_data['年份'].astype(str)
        
        merged = pd.merge(full_years, province_data, on='年份', how='left')
        
        # 特殊处理西藏数据
        if province == '西藏':
            merged['公共充电桩保有量（台）'] = merged['公共充电桩保有量（台）'].interpolate(method='linear')
        
        # 异常值处理
        if not merged['公共充电桩保有量（台）'].isnull().all():
            q_low = merged['公共充电桩保有量（台）'].quantile(0.05)
            q_high = merged['公共充电桩保有量（台）'].quantile(0.95)
            merged['公共充电桩保有量（台）'] = np.clip(
                merged['公共充电桩保有量（台）'], 
                q_low if not np.isnan(q_low) else 0, 
                q_high if not np.isnan(q_high) else merged['公共充电桩保有量（台）'].max()*2
            )
        
        # 填充可能的缺失值
        merged['公共充电桩保有量（台）'] = merged['公共充电桩保有量（台）'].fillna(method='ffill').fillna(method='bfill')
        
        # 创建时间序列
        dates = pd.date_range(start="2016-01-01", periods=7, freq='YS')
        return pd.Series(merged['公共充电桩保有量（台）'].values, index=dates)
    
    except Exception as e:
        print(f"预处理{province}数据时出错: {str(e)}")
        raise

def hybrid_forecast(series: pd.Series, province: str, steps: int = 8) -> np.ndarray:
    """混合预测模型"""
    min_growth, max_growth = (0.10, 0.50) if province in ['西藏', '青海', '宁夏'] else (0.05, 0.30)
    
    try:
        # 尝试ARIMA模型
        model = auto_arima(
            series,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            max_d=1,
            seasonal=False,
            suppress_warnings=True,
            error_action='ignore',
            trace=True
        )
        arima_model = ARIMA(series, order=model.order)
        arima_fit = arima_model.fit()
        
        if adfuller(arima_fit.resid)[1] > 0.05:
            raise ValueError("残差非白噪声")
            
        forecast = arima_fit.get_forecast(steps=steps).predicted_mean.values
        
    except Exception as arima_error:
        print(f"{province} ARIMA失败: {str(arima_error)}. 尝试Prophet...")
        
        try:
            # 准备Prophet数据
            prophet_df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            # 设置合理的容量上限
            cap = series.max() * 3 * PROVINCE_POLICY.get(province, 1.0)
            prophet_df['cap'] = cap
            
            # 训练Prophet模型
            model = Prophet(
                growth='logistic',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            model.add_country_holidays(country_name='CN')
            model.fit(prophet_df)
            
            # 生成预测
            future = model.make_future_dataframe(periods=steps, freq='YS')
            future['cap'] = cap * np.linspace(1, 1.5, len(future))
            forecast = model.predict(future)['yhat'].values[-steps:]
            
        except Exception as prophet_error:
            print(f"{province} Prophet失败: {str(prophet_error)}. 使用稳健增长...")
            base_value = series.iloc[-1]
            hist_growth = np.clip(series.pct_change().mean(), min_growth, max_growth)
            forecast = base_value * (1 + hist_growth) ** np.arange(1, steps+1)
    
    # 后处理
    max_cap = PROVINCE_CAPACITY.get(province, 1000000)
    forecast = np.clip(forecast, a_min=series.min()*0.9, a_max=max_cap)
    
    # 确保单调增长
    for i in range(1, len(forecast)):
        if forecast[i] < forecast[i-1]:
            forecast[i] = forecast[i-1] * (1 + min_growth)
    
    # 平滑处理
    return gaussian_filter1d(forecast, sigma=0.8)

def main():
    # 准备数据
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
    
    # 转换数据格式
    df = pd.DataFrame(data).melt(
        id_vars=["省份"],
        var_name="年份",
        value_name="公共充电桩保有量（台）"
    )
    
    # 准备空间权重
    adjacency = {
        "北京": ["天津", "河北"], "天津": ["北京", "河北"], 
        "河北": ["北京", "天津", "山西", "内蒙古", "辽宁", "山东", "河南"],
        "山西": ["河北", "内蒙古", "陕西", "河南"],
        "内蒙古": ["河北", "山西", "陕西", "宁夏", "甘肃", "黑龙江", "吉林", "辽宁"],
        "辽宁": ["河北", "内蒙古", "吉林"],
        "吉林": ["辽宁", "内蒙古", "黑龙江"],
        "黑龙江": ["吉林", "内蒙古"],
        "上海": ["江苏", "浙江"],
        "江苏": ["上海", "浙江", "安徽", "山东"],
        "浙江": ["上海", "江苏", "安徽", "江西", "福建"],
        "安徽": ["江苏", "浙江", "江西", "湖北", "河南", "山东"],
        "福建": ["浙江", "江西", "广东"],
        "江西": ["安徽", "浙江", "福建", "广东", "湖南", "湖北"],
        "山东": ["河北", "河南", "安徽", "江苏"],
        "河南": ["河北", "山西", "陕西", "湖北", "安徽", "山东"],
        "湖北": ["河南", "陕西", "重庆", "湖南", "江西", "安徽"],
        "湖南": ["湖北", "江西", "广东", "广西", "贵州", "重庆"],
        "广东": ["福建", "江西", "湖南", "广西", "海南"],
        "广西": ["湖南", "广东", "云南", "贵州"],
        "海南": ["广东"],
        "重庆": ["湖北", "湖南", "贵州", "四川", "陕西"],
        "四川": ["重庆", "陕西", "甘肃", "青海", "西藏", "云南", "贵州"],
        "贵州": ["重庆", "四川", "云南", "广西", "湖南"],
        "云南": ["四川", "贵州", "广西", "西藏"],
        "西藏": ["四川", "云南", "新疆", "青海"],
        "陕西": ["山西", "内蒙古", "宁夏", "甘肃", "四川", "重庆", "湖北", "河南"],
        "甘肃": ["内蒙古", "宁夏", "陕西", "四川", "青海", "新疆"],
        "青海": ["甘肃", "四川", "西藏", "新疆"],
        "宁夏": ["内蒙古", "陕西", "甘肃"],
        "新疆": ["甘肃", "青海", "西藏"]
    }
    
    # 确保省份顺序一致
    provinces = sorted(df['省份'].unique())
    w = prepare_spatial_weights(provinces, adjacency)
    
    # 准备各省份最新数据用于空间调整
    latest_data = df[df['年份'] == '2022'].set_index('省份')['公共充电桩保有量（台）'].to_dict()
    
    # 创建输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_dir, "预测结果")
    os.makedirs(output_folder, exist_ok=True)
    
    # 执行预测
    all_predictions = []
    for province in provinces:
        print(f"\n正在处理: {province}")
        province_data = df[df['省份'] == province]
        
        try:
            # 预处理数据
            series = preprocess_data(province_data, province)
            
            # 时间序列预测
            ts_forecast = hybrid_forecast(series, province)
            
            # 空间调整
            adjusted_forecast = [
                spatial_adjustment(
                    province=province,
                    value=value,
                    w=w,
                    province_data=latest_data
                )
                for value in ts_forecast
            ]
            
            all_predictions.append([province] + adjusted_forecast)
            
        except Exception as e:
            print(f"{province} 预测失败: {str(e)}")
            # 应急方案: 使用行业平均增长率12%
            base_value = province_data[province_data['年份'] == '2022']['公共充电桩保有量（台）'].values[0]
            forecast = [base_value * (1.12 ** i) for i in range(1, 9)]
            all_predictions.append([province] + forecast)
    
    # 保存结果
    if all_predictions:
        predictions_df = pd.DataFrame(
            all_predictions,
            columns=["省份"] + [f"{y}" for y in range(2023, 2031)]
        )
        output_path = os.path.join(output_folder, "最终预测结果.xlsx")
        predictions_df.to_excel(output_path, index=False)
        print(f"\n预测结果已保存至: {output_path}")

if __name__ == "__main__":
    main()