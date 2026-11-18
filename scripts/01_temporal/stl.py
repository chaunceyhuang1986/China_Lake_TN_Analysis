import os
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
import os

# -----------------------------
# 创建输出目录
# -----------------------------
def create_output_directory(output_path):
    try:
        os.makedirs(output_path, exist_ok=True)
        print(f"Output directory created/confirmed: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False

# -----------------------------
# 读取 TN 数据（year/month/day 格式）
# -----------------------------
def load_tn_data(data_file):
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return None

    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip()

    tn_col = [c for c in df.columns if 'tn' in c.lower()][0]

    # 合并 year/month/day 成 datetime
    df['date'] = pd.to_datetime(df[['year','month','day']], errors='coerce')

    # TN 列清理
    df[tn_col] = df[tn_col].astype(str).str.replace(',', '').str.strip()
    df[tn_col] = pd.to_numeric(df[tn_col], errors='coerce')

    # 删除缺失值
    df = df.dropna(subset=['date', tn_col])

    # 每日总体 TN（日均）
    daily_tn = df.groupby('date')[tn_col].mean().sort_index()

    # 过滤时间范围
    daily_tn = daily_tn[(daily_tn.index >= '2021-06-01') & (daily_tn.index <= '2024-12-31')]

    # 每年有效数据量
    yearly_counts = daily_tn.groupby(daily_tn.index.year).count()
    print("Daily points per year:")
    print(yearly_counts)
    print(f"Total points: {len(daily_tn)}")
    print(f"Date range: {daily_tn.index[0]} to {daily_tn.index[-1]}")
    print(f"TN range: {daily_tn.min():.4f} - {daily_tn.max():.4f} mg/L")

    return daily_tn

# -----------------------------
# STL 分解分析
# -----------------------------
def stl_decomposition_analysis(time_series, period=12, seasonal=13, robust=True):
    print("="*50)
    print("Starting STL Decomposition Analysis")
    print(f"Time series length: {len(time_series)}")
    print(f"Time range: {time_series.index[0]} to {time_series.index[-1]}")
    print(f"Seasonal period: {period}")
    print("="*50)

    stl = STL(time_series, period=period, seasonal=seasonal, robust=robust)
    result = stl.fit()

    print("\nSTL Decomposition Results Statistics:")
    print(f"Trend component range: [{result.trend.min():.4f}, {result.trend.max():.4f}]")
    print(f"Seasonal component range: [{result.seasonal.min():.4f}, {result.seasonal.max():.4f}]")
    print(f"Residual component range: [{result.resid.min():.4f}, {result.resid.max():.4f}]")
    print(f"Residual std: {result.resid.std():.4f}")

    return result

# -----------------------------
# 绘图函数
# -----------------------------
def plot_stl_results(result, output_dir):
    if not create_output_directory(output_dir):
        output_dir = "."

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.unicode_minus': False,
        'font.weight': 'bold',
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.labelsize': 20,
        'axes.titlesize': 20
    })

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.4, left=0.15, right=0.95, top=0.95, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # 过滤 2021-06 ~ 2024-12
    mask = (result.observed.index >= '2021-06-01') & (result.observed.index <= '2024-12-31')
    obs = result.observed[mask]
    trend = result.trend[mask]
    seasonal = result.seasonal[mask]
    resid = result.resid[mask]

    # Trend
    ax1.plot(trend.index, trend.values,'r-',linewidth=2)
    ax1.set_title('(d) Trend Component', fontsize=20,fontweight='bold',pad=15)
    ax1.set_ylabel('Trend Value (mg/L)', fontsize=20,fontweight='bold',labelpad=2)
    ax1.grid(True,alpha=0.3)
    ax1.set_xlim(obs.index[0], obs.index[-1])

    # Seasonal
    ax2.plot(seasonal.index, seasonal.values,'g-',linewidth=1.5)
    ax2.set_title('(e) Seasonal Component', fontsize=20,fontweight='bold',pad=15)
    ax2.set_ylabel('Seasonal Variation (mg/L)', fontsize=20,fontweight='bold',labelpad=2)
    ax2.grid(True,alpha=0.3)
    ax2.set_xlim(obs.index[0], obs.index[-1])

    # Residual
    ax3.plot(resid.index, resid.values,'purple',linewidth=1)
    ax3.set_title('(f) Residual Component', fontsize=20,fontweight='bold',pad=15)
    ax3.set_xlabel('Time', fontsize=20,fontweight='bold',labelpad=10)
    ax3.set_ylabel('Residual Value (mg/L)', fontsize=20,fontweight='bold',labelpad=2)
    ax3.grid(True,alpha=0.3)
    ax3.set_xlim(obs.index[0], obs.index[-1])

    # 保存图片
    output_file = os.path.join(output_dir,'Total_Nitrogen_STL_Decomposition_2021_2024.png')
    plt.savefig(output_file,dpi=300,bbox_inches='tight',facecolor='white')
    print(f"STL decomposition plot saved as: {output_file}")
    plt.show()

    return fig, output_file

# -----------------------------
# 保存 STL 结果 CSV
# -----------------------------
def save_decomposition_results(result, output_dir, original_file_path):
    create_output_directory(output_dir)
    decomposition_df = pd.DataFrame({
        'date': result.observed.index,
        'observed': result.observed.values,
        'trend': result.trend.values,
        'seasonal': result.seasonal.values,
        'residual': result.resid.values
    })

    # 保存到 STL 文件夹
    output_file = os.path.join(output_dir,'Total_Nitrogen_STL_Decomposition_2021_2024.csv')
    decomposition_df.to_csv(output_file,index=False,encoding='utf-8')
    print(f"Decomposition results saved as: {output_file}")

    # 保存到原始 CSV 所在文件夹
    original_folder = os.path.dirname(original_file_path)
    original_csv_file = os.path.join(original_folder,'Total_Nitrogen_STL_Decomposition_2021_2024.csv')
    decomposition_df.to_csv(original_csv_file,index=False,encoding='utf-8')
    print(f"Decomposition results also saved to original folder: {original_csv_file}")

    return output_file, original_csv_file

# -----------------------------
# 主函数
# -----------------------------
def main(data_file,r_output_dir):
    ts = load_tn_data(data_file)
    if ts is None:
        print("No valid data found!")
        return

    stl_result = stl_decomposition_analysis(ts, period=365, seasonal=367, robust=True)
    fig, img_file = plot_stl_results(stl_result, r_output_dir)
    csv_file, csv_file_orig = save_decomposition_results(stl_result, r_output_dir, data_file)

    print("\nSTL Decomposition Completed!")
    print(f"Plot file: {img_file}")
    print(f"CSV file (STL folder): {csv_file}")
    print(f"CSV file (original folder): {csv_file_orig}")

    return fig, stl_result, img_file, csv_file, csv_file_orig

# -----------------------------
# 执行
# -----------------------------
if __name__ == "__main__":
    data_file = os.environ.get("TN_DAILY_INPUT", "data/private/lake_TN_daily_2021_2024.csv")
    output_dir = os.environ.get("TN_OUTPUT_DIR", "outputs/stl")
    main(data_file, output_dir)