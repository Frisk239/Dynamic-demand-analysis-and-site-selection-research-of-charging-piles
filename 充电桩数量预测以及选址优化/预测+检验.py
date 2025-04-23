import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据准备 ====================
data = {
    'Year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Sales': [73, 125, 211, 310, 410, 520, 630, 958, 1316],
    'Patents': [8657, 11985, 19506, 18236, 27382, 35104, 40837, 40956, 49250],
    'Chargers': [14.1254, 21.3903, 29.9749, 51.6211, 80.7383, 114.5380, 179.5674, 201.1155, 225.2493]
}
df = pd.DataFrame(data)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)
df = df.asfreq('YS')

# ==================== 模型定义 ====================
def prophet_forecast(df, col, periods=6):
    df_p = df[[col]].reset_index().rename(columns={'Year':'ds', col:'y'})
    model = Prophet(seasonality_mode='multiplicative',
                   changepoint_prior_scale=0.5,
                   yearly_seasonality=False)
    model.fit(df_p)
    future = model.make_future_dataframe(periods=periods, freq='YS')
    forecast = model.predict(future)
    return forecast.set_index('ds')[['yhat']].rename(columns={'yhat': f'{col}_Prophet'})

def var_forecast(df):
    df_diff = df.diff().dropna()
    model = VAR(df_diff)
    result = model.fit(maxlags=3)
    forecast = result.forecast(df_diff.values[-3:], steps=6)
    
    # 还原差分
    last_values = df.iloc[-1].values
    forecast_values = []
    for i in range(6):
        last_values += forecast[i]
        forecast_values.append(last_values.copy())
    
    forecast_index = pd.date_range(start='2025-01-01', periods=6, freq='YS')
    return pd.DataFrame(forecast_values, index=forecast_index, 
                       columns=[f'{col}_VAR' for col in df.columns])

def create_features(df):
    df_feat = df.copy()
    for lag in range(1, 3):
        for col in df.columns:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
    df_feat['time_index'] = np.arange(len(df_feat))
    df_feat['Sales_ma3'] = df_feat['Sales'].rolling(3).mean()
    return df_feat.dropna()

# ==================== 预测执行 ====================
prophet_preds = []
for col in ['Sales', 'Patents', 'Chargers']:
    prophet_preds.append(prophet_forecast(df, col))
prophet_result = pd.concat(prophet_preds, axis=1)

var_forecast_df = var_forecast(df)

rf_df = create_features(df)
X = rf_df.drop(['Sales', 'Patents', 'Chargers'], axis=1)
y = rf_df[['Sales', 'Patents', 'Chargers']]

rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X, y)

forecast_index = pd.date_range(start='2025-01-01', periods=6, freq='YS')
future_rf = pd.DataFrame(index=forecast_index, columns=['Sales_RF', 'Patents_RF', 'Chargers_RF'])
last_row = X.iloc[-1].copy()

for i in range(6):
    pred = rf.predict([last_row])[0]
    future_rf.iloc[i] = pred
    for j, var in enumerate(['Sales', 'Patents', 'Chargers']):
        last_row[f'{var}_lag2'] = last_row[f'{var}_lag1']
        last_row[f'{var}_lag1'] = pred[j]
    last_row['time_index'] += 1

result_df = pd.concat([
    prophet_result[['Sales_Prophet', 'Patents_Prophet', 'Chargers_Prophet']],
    var_forecast_df,
    future_rf
], axis=1)

# ==================== 绘图函数 ====================
def plot_forecast(var_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[var_name], label='历史数据', marker='o', color='black', linewidth=2)
    
    last_actual = df[var_name].iloc[-1]
    forecast_df = result_df[[f'{var_name}_Prophet', f'{var_name}_VAR', f'{var_name}_RF']]
    forecast_df = forecast_df[forecast_df.index >= '2025-01-01']
    
    all_dates = [df.index[-1]] + forecast_df.index.tolist()
    
    plt.plot(all_dates, [last_actual] + forecast_df[f'{var_name}_Prophet'].tolist(),
             linestyle='--', marker='s', color='blue', alpha=0.7, label='Prophet预测')
    plt.plot(all_dates, [last_actual] + forecast_df[f'{var_name}_VAR'].tolist(),
             linestyle='--', marker='^', color='green', alpha=0.7, label='VAR预测')
    plt.plot(all_dates, [last_actual] + forecast_df[f'{var_name}_RF'].tolist(),
             linestyle='--', marker='D', color='red', alpha=0.7, label='随机森林预测')

    plt.title(f'{var_name}预测（2016-2030）', fontsize=14)
    plt.xlabel('年份')
    plt.ylabel(var_name)
    plt.xticks(pd.date_range(start='2016', end='2031', freq='YS'), 
              [year.strftime('%Y') for year in pd.date_range(start='2016', end='2031', freq='YS')], 
              rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==================== 模型检测 ====================
def model_health_check():
    print("\n" + "="*50)
    print("模型健康检测报告（基于MAPE指标）")
    print("="*50)
    
    test_data = df.loc['2022':'2024']
    train_data = df.loc[:'2021']
    
    # 存储各模型预测结果用于绘图
    pred_results = {}
    
    def evaluate_model(model_type, col='Sales'):
        try:
            if model_type == 'Prophet':
                m = Prophet(seasonality_mode='multiplicative')
                m.fit(train_data[[col]].reset_index().rename(columns={'Year':'ds', col:'y'}))
                future = m.make_future_dataframe(periods=3, freq='YS')
                pred = m.predict(future).set_index('ds')['yhat'].iloc[-3:]
            
            elif model_type == 'VAR':
                df_diff = train_data.diff().dropna()
                m = VAR(df_diff)
                res = m.fit(maxlags=3)
                pred_diff = res.forecast(df_diff.values[-3:], steps=3)[:, df_diff.columns.get_loc(col)]
                pred = train_data[col].iloc[-1] + np.cumsum(pred_diff)
                pred = pd.Series(pred, index=test_data.index)
            
            elif model_type == 'RF':
                X_train = train_data.index.astype(int).values.reshape(-1,1)
                y_train = train_data[col].values
                m = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
                m.fit(X_train, y_train)
                pred = m.predict(test_data.index.astype(int).values.reshape(-1,1))
                pred = pd.Series(pred, index=test_data.index)
            
            mape = mean_absolute_percentage_error(test_data[col], pred)
            return mape < 20, f"MAPE={mape:.1f}%", pred
        
        except Exception as e:
            return False, f"运行失败: {str(e)}", None
    
    # 检测三个模型
    models = ['Prophet', 'VAR', 'RF']
    colors = ['blue', 'green', 'red']
    plt.figure(figsize=(10,4))
    plt.plot(test_data['Sales'], label='实际值', marker='o', color='black')
    
    for model, color in zip(models, colors):
        status, detail, pred = evaluate_model(model)
        print(f"{model}模型: {'✅通过' if status else '❌失败'} | {detail}")
        if status and pred is not None:
            pred_results[model] = pred
            plt.plot(pred, label=f'{model}预测', linestyle='--', color=color)
    
    plt.title('验证期(2022-2024)预测对比')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n检测结论（MAPE<20%为通过）:")
    for model in models:
        status, detail, _ = evaluate_model(model)
        print(f"- {model}: {'正常' if status else '异常'} ({detail})")
    print("="*50 + "\n")

# ==================== 主程序 ====================
if __name__ == '__main__':
    plot_forecast('Sales')
    plot_forecast('Patents')
    plot_forecast('Chargers')
    model_health_check()