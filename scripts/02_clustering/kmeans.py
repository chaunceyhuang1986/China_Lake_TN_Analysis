import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(data_path):
    """加载和预处理水质数据"""
    data_file = Path(data_path) / "processed_water_quality.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    # 读取数据
    df = pd.read_csv(data_file)
    print(f"成功加载数据，总记录数: {len(df)}")

    # 重命名列以保持一致性
    df = df.rename(columns={
        'Hylak_id': 'hylak_id',
        'Year': 'year',
        'Month': 'month',
        'TN': 'tn'
    })

    # 构建完整日期
    df['Date'] = pd.to_datetime({
        'year': df['year'],
        'month': df['month'],
        'day': 15
    })

    df = df[df['Date'].dt.year.between(2021, 2024)].copy()
    if df.empty:
        raise ValueError('No observations available for 2021-2024')
    df = df.sort_values(['hylak_id', 'Date'])

    print(f"总数据量: {len(df)} 条记录")
    print(f"涉及湖泊数量: {df['hylak_id'].nunique()} 个")
    print(f"时间范围: {df['Date'].min()} 到 {df['Date'].max()}")
    print(f"总氮数据范围: {df['tn'].min():.3f} - {df['tn'].max():.3f} mg/L")

    return df


def calculate_tn_features(df, split_year=2022):
    """2021–2024 TN features; default comparison: 2021–2022 vs 2023–2024."""
    if not 2021 <= split_year < 2024:
        raise ValueError('split_year must be between 2021 and 2023')
    df = df[df['Date'].dt.year.between(2021, 2024)].copy()
    features = []

    for lake_id, group in df.groupby('hylak_id'):
        # 确保有足够的数据点
        if len(group) < 6:
            continue

        # 划分时段
        p1 = group[group['Date'].dt.year <= split_year]
        p2 = group[group['Date'].dt.year > split_year]

        if len(p1) == 0 or len(p2) == 0:
            continue

        # 计算总氮特征
        tn_p1_mean = p1['tn'].mean()
        tn_p2_mean = p2['tn'].mean()
        tn_change = tn_p2_mean - tn_p1_mean
        tn_change_rate = (tn_change / tn_p1_mean) * 100 if tn_p1_mean > 0 else 0

        features.append({
            'hylak_id': lake_id,
            'Period1_TN_Mean': tn_p1_mean,
            'Period2_TN_Mean': tn_p2_mean,
            'TN_Change': tn_change,
            'TN_Change_Rate': tn_change_rate,
            'Data_Points_P1': len(p1),
            'Data_Points_P2': len(p2),
            'Total_Points': len(group)
        })

    if len(features) == 0:
        raise ValueError("没有找到足够数据的湖泊进行分析")

    feature_df = pd.DataFrame(features)
    print(f"有效湖泊数量: {len(feature_df)}")

    return feature_df


def uniform_kmeans_clustering(feature_df, target_clusters=4):
    """强制均匀分布的K-means聚类"""
    print(f"执行均匀分布聚类，目标聚类数: {target_clusters}")

    # 准备数据
    X = feature_df[['Period1_TN_Mean', 'Period2_TN_Mean', 'TN_Change']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 第一步：使用K-means进行初始聚类（稍多一些聚类）
    initial_clusters = min(target_clusters * 2, len(feature_df) // 3)
    kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=20)
    initial_labels = kmeans.fit_predict(X_scaled)

    # 第二步：基于特征值进行分层，确保均匀分布
    feature_df['Initial_Cluster'] = initial_labels

    # 按初始聚类计算特征统计
    cluster_stats = feature_df.groupby('Initial_Cluster').agg({
        'Period1_TN_Mean': 'mean',
        'Period2_TN_Mean': 'mean',
        'TN_Change': 'mean',
        'hylak_id': 'count'
    }).rename(columns={'hylak_id': 'count'})

    # 第三步：使用分位数方法强制均匀分布
    # 基于前期浓度进行分层
    feature_df_sorted = feature_df.sort_values('Period1_TN_Mean')
    feature_df_sorted['Uniform_Cluster'] = pd.cut(
        feature_df_sorted.index,
        bins=target_clusters,
        labels=range(target_clusters)
    )

    # 第四步：使用分层结果作为K-means的初始中心点
    cluster_centers = []
    for cluster_id in range(target_clusters):
        cluster_data = feature_df_sorted[feature_df_sorted['Uniform_Cluster'] == cluster_id]
        if len(cluster_data) > 0:
            center = cluster_data[['Period1_TN_Mean', 'Period2_TN_Mean', 'TN_Change']].mean().values
            cluster_centers.append(center)

    # 标准化中心点
    cluster_centers = scaler.transform(np.array(cluster_centers))

    # 最终K-means聚类
    kmeans_final = KMeans(
        n_clusters=target_clusters,
        init=cluster_centers,
        random_state=42,
        n_init=1
    )
    final_labels = kmeans_final.fit_predict(X_scaled)

    result_df = feature_df.copy()
    result_df['Cluster'] = final_labels
    result_df['Method'] = f'Uniform_KMeans_{target_clusters}'

    return result_df


def evaluate_clustering_quality(result_df, X_scaled):
    """评估聚类质量"""
    cluster_sizes = result_df['Cluster'].value_counts().sort_index()
    n_clusters = len(cluster_sizes)

    # 计算质量指标
    silhouette_avg = silhouette_score(X_scaled, result_df['Cluster'])

    # 计算均匀度（标准差越小越均匀）
    size_std = cluster_sizes.std()
    size_mean = cluster_sizes.mean()
    uniformity = 1 - (size_std / size_mean) if size_mean > 0 else 0

    quality_metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'uniformity': uniformity,
        'cluster_sizes': cluster_sizes.to_dict(),
        'size_std': size_std,
        'size_mean': size_mean
    }

    return quality_metrics


def find_optimal_uniform_clustering(feature_df):
    """寻找最优的均匀聚类方案"""
    X = feature_df[['Period1_TN_Mean', 'Period2_TN_Mean', 'TN_Change']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_score = -1
    best_result = None
    best_metrics = None
    best_n_clusters = 3

    # 尝试不同的聚类数量（3-6个）
    for n_clusters in range(3, 7):
        if n_clusters >= len(feature_df) // 3:
            continue

        print(f"尝试 {n_clusters} 类聚类...")
        result_df = uniform_kmeans_clustering(feature_df, n_clusters)
        metrics = evaluate_clustering_quality(result_df, X_scaled)

        # 综合评分（侧重均匀性）
        score = 0.6 * metrics['uniformity'] + 0.4 * metrics['silhouette_score']

        print(f"  聚类数: {n_clusters}, 均匀度: {metrics['uniformity']:.3f}, "
              f"轮廓系数: {metrics['silhouette_score']:.3f}, 综合得分: {score:.3f}")

        if score > best_score:
            best_score = score
            best_result = result_df
            best_metrics = metrics
            best_n_clusters = n_clusters

    return best_result, best_metrics


def create_improved_visualization(result_df, output_path):
    """创建改进的可视化图表（精确控制字体大小）"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # 进一步增加画布尺寸

    # 设置具体的字体大小
    title_fontsize = 24  # 标题字体大小
    label_fontsize = 22 # 坐标轴标签字体大小
    tick_fontsize = 20  # 刻度标签字体大小
    legend_fontsize = 22  # 图例字体大小
    pie_fontsize = 22  # 饼图字体大小

    # 1. 聚类分布饼图
    cluster_counts = result_df['Cluster'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))

    axes[0, 0].pie(cluster_counts.values, labels=[f'聚类{i}' for i in cluster_counts.index],
                   autopct='%1.1f%%', startangle=90, colors=colors,
                   textprops={'fontsize': pie_fontsize})
    axes[0, 0].set_title('聚类分布', fontsize=title_fontsize, fontweight='bold')

    # 2. 前后期浓度对比散点图
    for i, cluster_id in enumerate(cluster_counts.index):
        cluster_data = result_df[result_df['Cluster'] == cluster_id]
        axes[0, 1].scatter(cluster_data['Period1_TN_Mean'], cluster_data['Period2_TN_Mean'],
                           c=[colors[i]], label=f'聚类{cluster_id}', alpha=0.7, s=80)  # 增加点的大小

    # 添加对角线
    min_val = min(result_df[['Period1_TN_Mean', 'Period2_TN_Mean']].min())
    max_val = max(result_df[['Period1_TN_Mean', 'Period2_TN_Mean']].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='无变化线', linewidth=2)
    axes[0, 1].set_xlabel('前期平均总氮浓度 (mg/L)', fontsize=label_fontsize)
    axes[0, 1].set_ylabel('后期平均总氮浓度 (mg/L)', fontsize=label_fontsize)
    axes[0, 1].set_title('前后期总氮浓度对比', fontsize=title_fontsize, fontweight='bold')
    axes[0, 1].legend(fontsize=legend_fontsize)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # 3. 总氮变化率箱线图
    change_data = []
    labels = []
    for cluster_id in sorted(result_df['Cluster'].unique()):
        cluster_data = result_df[result_df['Cluster'] == cluster_id]
        change_data.append(cluster_data['TN_Change_Rate'])
        labels.append(f'聚类{cluster_id}')

    box_plot = axes[1, 0].boxplot(change_data, labels=labels, patch_artist=True)
    # 设置箱线图颜色和线宽
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(box_plot[element], linewidth=2)

    axes[1, 0].set_ylabel('总氮变化率 (%)', fontsize=label_fontsize)
    axes[1, 0].set_title('各聚类总氮变化率分布', fontsize=title_fontsize, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=tick_fontsize)

    # 4. 聚类特征条形图
    cluster_means = result_df.groupby('Cluster').agg({
        'Period1_TN_Mean': 'mean',
        'Period2_TN_Mean': 'mean',
        'TN_Change': 'mean'
    })

    x = np.arange(len(cluster_means))
    width = 0.25

    bars1 = axes[1, 1].bar(x - width, cluster_means['Period1_TN_Mean'], width,
                           label='前期浓度', alpha=0.7, color='skyblue')
    bars2 = axes[1, 1].bar(x, cluster_means['Period2_TN_Mean'], width,
                           label='后期浓度', alpha=0.7, color='lightcoral')
    bars3 = axes[1, 1].bar(x + width, cluster_means['TN_Change'], width,
                           label='变化量', alpha=0.7, color='lightgreen')

    axes[1, 1].set_xlabel('聚类编号', fontsize=label_fontsize)
    axes[1, 1].set_ylabel('浓度值 (mg/L)', fontsize=label_fontsize)
    axes[1, 1].set_title('各聚类特征对比', fontsize=title_fontsize, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'聚类{i}' for i in cluster_means.index], fontsize=tick_fontsize)
    axes[1, 1].legend(fontsize=legend_fontsize)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # 调整子图间距
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_path, '改进的总氮聚类分析结果_大字体.png'), dpi=300, bbox_inches='tight')
    plt.show()


def save_improved_results(result_df, metrics, output_path):
    """保存改进的分析结果"""
    # 保存详细结果
    detailed_output = os.path.join(output_path, '改进的总氮聚类详细结果.csv')
    result_df.to_csv(detailed_output, index=False, encoding='utf-8-sig')

    # 生成分析报告
    report_output = os.path.join(output_path, '改进的总氮聚类分析报告.txt')
    with open(report_output, 'w', encoding='utf-8') as f:
        f.write("=== 湖泊总氮(TN)改进聚类分析报告 ===\n\n")
        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析湖泊数量: {len(result_df)} 个\n")
        f.write(f"最终聚类数量: {metrics['n_clusters']} 个\n")
        f.write(f"轮廓系数: {metrics['silhouette_score']:.3f}\n")
        f.write(f"均匀度: {metrics['uniformity']:.3f}\n")
        f.write(f"聚类大小标准差: {metrics['size_std']:.1f}\n\n")

        f.write("=== 各聚类详细统计 ===\n")
        for cluster_id in sorted(result_df['Cluster'].unique()):
            cluster_data = result_df[result_df['Cluster'] == cluster_id]
            count = len(cluster_data)
            percentage = count / len(result_df) * 100

            f.write(f"\n聚类{cluster_id} ({count}个湖泊, {percentage:.1f}%):\n")
            f.write(
                f"  前期平均浓度: {cluster_data['Period1_TN_Mean'].mean():.3f} ± {cluster_data['Period1_TN_Mean'].std():.3f} mg/L\n")
            f.write(
                f"  后期平均浓度: {cluster_data['Period2_TN_Mean'].mean():.3f} ± {cluster_data['Period2_TN_Mean'].std():.3f} mg/L\n")
            f.write(f"  平均变化量: {cluster_data['TN_Change'].mean():.3f} mg/L\n")
            f.write(f"  平均变化率: {cluster_data['TN_Change_Rate'].mean():.1f}%\n")

            # 判断变化趋势
            avg_change = cluster_data['TN_Change'].mean()
            if avg_change > 0.1:
                trend = "显著上升"
            elif avg_change < -0.1:
                trend = "显著下降"
            else:
                trend = "基本稳定"
            f.write(f"  变化趋势: {trend}\n")


def improved_tn_cluster_analysis():
    """改进的总氮聚类分析主函数"""
    # 设置路径
    data_path = os.environ.get("TN_CLUSTER_INPUT_DIR", "data/private")
    output_path = os.environ.get("TN_OUTPUT_DIR", "outputs/clustering")

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    try:
        # 1. 加载和预处理数据
        print("正在加载数据...")
        df = load_and_preprocess_data(data_path)

        # 2. 计算总氮特征
        print("计算总氮特征...")
        feature_df = calculate_tn_features(df)

        # 3. 寻找最优均匀聚类方案
        print("\n寻找最优均匀聚类方案...")
        best_result, best_metrics = find_optimal_uniform_clustering(feature_df)

        # 4. 显示结果统计
        print(f"\n=== 最优聚类方案 ===")
        print(f"聚类数量: {best_metrics['n_clusters']}")
        print(f"轮廓系数: {best_metrics['silhouette_score']:.3f}")
        print(f"均匀度: {best_metrics['uniformity']:.3f}")

        cluster_stats = best_result['Cluster'].value_counts().sort_index()
        print("\n聚类分布:")
        for cluster_id, count in cluster_stats.items():
            percentage = count / len(best_result) * 100
            print(f"  聚类{cluster_id}: {count}个湖泊 ({percentage:.1f}%)")

        # 5. 可视化结果
        print("生成可视化图表...")
        create_improved_visualization(best_result, output_path)

        # 6. 保存结果
        print("保存分析结果...")
        save_improved_results(best_result, best_metrics, output_path)

        print(f"\n=== 分析完成 ===")
        print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


# 执行分析
if __name__ == "__main__":
    improved_tn_cluster_analysis()
