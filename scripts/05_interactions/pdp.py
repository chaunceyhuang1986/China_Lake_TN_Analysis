import os
import os
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects
from matplotlib.ticker import FormatStrFormatter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# 全局设置
# =========================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.unicode_minus': False
})

sns.set_theme(font='Times New Roman')
warnings.filterwarnings('ignore')

output_dir = os.environ.get("TN_OUTPUT_DIR", "outputs/pdp")
os.makedirs(output_dir, exist_ok=True)

data_file = os.environ.get("TN_MERGED_INPUT", "data/private/merged_lake_data_complete.csv")


# =========================
# 标签显示名称映射
# 只改图上的显示，不改真实列名
# =========================
DISPLAY_NAME_MAP = {
    'min_tem': 'Minimum Air Temperature',
    'PLAND-Water': 'PLAND-Water',
    'Lake surface area (km²)': 'Lake surface area (km²)',
    'HFI': 'HFI',
    'PLAND-Cropland': 'PLAND-Cropland',
    'PLAND-Grassland': 'PLAND-Grassland'
}


def pretty_name(name):
    key = str(name).strip()
    return DISPLAY_NAME_MAP.get(key, name)


# =========================
# 工具函数
# =========================
def clean_filename(filename):
    """清理文件名中的特殊字符"""
    invalid_chars = '<>:"/\\|?*'
    filename = str(filename)
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.replace(' ', '_')


def set_axis_one_decimal(ax):
    """坐标轴刻度统一保留 1 位小数"""
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def set_colorbar_one_decimal(cbar):
    """颜色条刻度统一保留 1 位小数"""
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar.ax.tick_params(labelsize=20, colors='black')
    cbar.update_ticks()


def style_axis(ax, tick_size=28, spine_width=3.0):
    """统一坐标轴风格"""
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=tick_size,
        width=spine_width,
        length=8,
        direction='in',
        bottom=True,
        top=True,
        left=True,
        right=True,
        colors='black',
        labelcolor='black',
        pad=10
    )

    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
        label.set_color('black')

    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
        label.set_color('black')

    ax.xaxis.label.set_fontname('Times New Roman')
    ax.yaxis.label.set_fontname('Times New Roman')
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
        spine.set_visible(True)
        spine.set_color('black')


# =========================
# 读取数据
# =========================
print("正在读取数据...")
data = pd.read_csv(data_file)
print(f"数据形状: {data.shape}")

# =========================
# 面积单位转换：m² -> km²
# 真实数据列不删，新增一个用于建模和绘图的列
# =========================
if 'Lake surface area' in data.columns:
    data['Lake surface area (km²)'] = data['Lake surface area'] / 1_000_000
    print("已将 'Lake surface area' 从 m² 转换为 'Lake surface area (km²)'")
elif 'Lake surface area (km²)' in data.columns:
    print("检测到 'Lake surface area (km²)' 已存在，跳过转换")
else:
    print("警告：未找到 'Lake surface area' 列，也未找到 'Lake surface area (km²)' 列")

# =========================
# 自动查找目标变量
# =========================
print("\n" + "=" * 60)
print("查找总氮目标变量")
print("=" * 60)

print("数据列名:")
for i, col in enumerate(data.columns):
    print(f"{i:3d}. {col}")

total_n_keywords = ['total_n', 'total nitrogen', 'TN', 'n_total', 'nitrogen']
found_targets = []

for col in data.columns:
    col_lower = col.lower()
    for keyword in total_n_keywords:
        if keyword in col_lower or col_lower in keyword:
            found_targets.append(col)
            break

if 'TN' in data.columns and 'TN' not in found_targets:
    found_targets.append('TN')

if found_targets:
    target_col = found_targets[0]
    print(f"\n自动选择目标变量: '{target_col}' (索引: {list(data.columns).index(target_col)})")
else:
    print("\n未找到总氮相关列，请手动输入列索引:")
    while True:
        try:
            user_input = input("请输入列索引: ").strip()
            if '.' in user_input:
                user_input = user_input.split('.')[0].strip()

            col_idx = int(user_input)
            if 0 <= col_idx < len(data.columns):
                target_col = data.columns[col_idx]
                print(f"手动选择目标变量: '{target_col}' (索引: {col_idx})")
                break
            else:
                print(f"错误: 索引必须在 0-{len(data.columns) - 1} 之间")
        except ValueError:
            print("错误: 请输入有效数字")

print(f"\n最终目标变量: {target_col}")

# =========================
# 定义关键因子
# 注意：这里用真实列名 min_tem
# =========================
key_features = [
    'min_tem',
    'PLAND-Water',
    'Lake surface area (km²)',
    'HFI',
    'PLAND-Cropland',
    'PLAND-Grassland'
]

available_features = []
missing_features = []

for feature in key_features:
    if feature in data.columns:
        available_features.append(feature)
    else:
        missing_features.append(feature)
        possible_variants = [
            feature.replace('-', ' '),
            feature.replace('-', '_'),
            feature.upper(),
            feature.lower(),
        ]
        for variant in possible_variants:
            if variant in data.columns:
                available_features.append(variant)
                missing_features.remove(feature)
                print(f"找到变体: '{variant}' 替代 '{feature}'")
                break

print(f"\n使用的关键因子 ({len(available_features)}个):")
for i, feature in enumerate(available_features):
    print(f"  {i + 1:2d}. {pretty_name(feature):30s} (原列名: {feature})")

if missing_features:
    print(f"\n警告: 以下特征未找到: {missing_features}")

if len(available_features) < 2:
    raise ValueError(f"至少需要 2 个关键因子，只找到 {len(available_features)} 个")

# =========================
# 分离特征和目标变量
# =========================
X = data[available_features].copy()
y = data[target_col].copy()

print(f"\n" + "=" * 60)
print("数据统计信息")
print("=" * 60)
print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"目标变量: {target_col}")
print(f"目标变量统计:")
print(f"  均值: {y.mean():.4f}")
print(f"  标准差: {y.std():.4f}")
print(f"  最小值: {y.min():.4f}")
print(f"  最大值: {y.max():.4f}")
print(f"  中位数: {y.median():.4f}")

# =========================
# 缺失值处理
# =========================
print("\n" + "=" * 60)
print("处理缺失值")
print("=" * 60)

missing_y = y.isnull().sum()
missing_X = X.isnull().sum().sum()

if missing_y > 0 or missing_X > 0:
    print(f"目标变量缺失值: {missing_y}")
    print(f"特征缺失值总数: {missing_X}")

    if missing_y > 0:
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"删除目标变量缺失后数据形状: {X.shape}")

    if X.isnull().sum().sum() > 0:
        print("使用 KNN 插补处理特征缺失值...")
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        print("KNN 插补完成")
else:
    print("数据中没有缺失值")

print(f"\n最终数据形状: {X.shape}")
print(f"特征缺失值总数: {X.isnull().sum().sum()}")
print(f"目标变量缺失值总数: {y.isnull().sum()}")

# =========================
# 标准化
# =========================
print("\n" + "=" * 60)
print("数据标准化")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("特征标准化完成")

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
print("目标变量标准化完成")

# =========================
# 数据集划分
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

print(f"\n训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# =========================
# 训练模型
# =========================
print("\n" + "=" * 60)
print("训练梯度提升回归模型")
print("=" * 60)

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    random_state=42
)

print("开始训练模型...")
gbr.fit(X_train, y_train)
print("模型训练完成")

# =========================
# 预测和评估
# =========================
y_pred_scaled = gbr.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print("\n模型性能:")
print("均方误差 (MSE):", mse)
print("R² 分数:", r2)

# =========================
# 特征重要性
# =========================
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gbr.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n关键因子重要性排序:")
for i, row in enumerate(feature_importance.itertuples(), 1):
    print(f"{i:2d}. {pretty_name(row.Feature):30s}: {row.Importance:.4f}")

importance_path = os.path.join(output_dir, "关键因子重要性.csv")
feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
print(f"\n特征重要性已保存到: {importance_path}")

# =========================
# 清理文件名中的特殊字符
# =========================
def clean_filename(filename):
    invalid_chars = '<>:"/\\|?*'
    filename = str(filename)
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.replace(' ', '_')

# =========================
# 生成双变量PDP图
# =========================
print("\n" + "=" * 60)
print("生成偏依赖图")
print("=" * 60)

n_features = len(X.columns)
features = X.columns
interaction_matrix = np.zeros((n_features, n_features))
interaction_range_matrix = np.zeros((n_features, n_features))
interaction_std_matrix = np.zeros((n_features, n_features))

all_pairs = [(feature_1, feature_2) for i, feature_1 in enumerate(features)
             for feature_2 in features[i + 1:]]

print(f"将生成 {len(all_pairs)} 个双变量交互图")

success_count = 0
failed_pairs = []

for i, (feature_1, feature_2) in enumerate(all_pairs):
    try:
        # 强制 Lake surface area 在 y 轴
        if feature_1 == 'Lake surface area (km²)':
            feature_1, feature_2 = feature_2, feature_1

        clean_f1 = clean_filename(pretty_name(feature_1))
        clean_f2 = clean_filename(pretty_name(feature_2))

        index_1 = features.get_loc(feature_1)
        index_2 = features.get_loc(feature_2)

        pdp_display = PartialDependenceDisplay.from_estimator(
            gbr,
            X_scaled,
            [(index_1, index_2)],
            grid_resolution=10,
            random_state=42
        )

        XX, YY = np.meshgrid(
            pdp_display.pd_results[0].grid_values[0],
            pdp_display.pd_results[0].grid_values[1]
        )
        Z = pdp_display.pd_results[0].average.reshape(XX.shape)

        XX_original = XX * scaler.scale_[index_1] + scaler.mean_[index_1]
        YY_original = YY * scaler.scale_[index_2] + scaler.mean_[index_2]
        Z_original = y_scaler.inverse_transform(Z.reshape(-1, 1)).reshape(Z.shape)

        z_range = Z_original.max() - Z_original.min()
        z_std = np.std(Z_original)
        z_mean = np.mean(np.abs(Z_original))

        interaction_matrix[index_1, index_2] = z_mean
        interaction_matrix[index_2, index_1] = z_mean
        interaction_range_matrix[index_1, index_2] = z_range
        interaction_range_matrix[index_2, index_1] = z_range
        interaction_std_matrix[index_1, index_2] = z_std
        interaction_std_matrix[index_2, index_1] = z_std

        fig, ax = plt.subplots(figsize=(12, 10))

        min_z, max_z = Z_original.min(), Z_original.max()
        if max_z - min_z < 1e-10:
            plt.close(fig)
            continue

        x_min, x_max = XX_original.min(), XX_original.max()
        y_min, y_max = YY_original.min(), YY_original.max()

        contour_levels = np.linspace(min_z, max_z, 10)

        contour = ax.contourf(
            XX_original,
            YY_original,
            Z_original,
            levels=contour_levels,
            cmap="viridis",
            alpha=0.85
        )

        contour_lines = ax.contour(
            XX_original,
            YY_original,
            Z_original,
            levels=contour_levels,
            colors='darkblue',
            linewidths=1.0,
            alpha=0.7
        )

        ax.clabel(
            contour_lines,
            inline=True,
            fontsize=20,
            colors='black',
            fmt='%.1f',
            levels=contour_lines.levels[::2],
            inline_spacing=6,
            manual=False,
            use_clabeltext=True
        )

        for text in contour_lines.labelTexts:
            text.set_fontname('Times New Roman')
            text.set_fontsize(20)
            text.set_fontweight('bold')
            text.set_path_effects([
                patheffects.withStroke(linewidth=2, foreground='white')
            ])

        cbar = fig.colorbar(contour, ax=ax, pad=0.02, aspect=20, shrink=0.8)
        cbar.set_label(
            'Partial Dependence',
            fontsize=24,
            weight='bold',
            fontname='Times New Roman',
            color='black'
        )
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)
            label.set_fontweight('bold')
            label.set_color('black')
        set_colorbar_one_decimal(cbar)

        ax.set_xlabel(
            pretty_name(feature_1),
            fontsize=38,
            weight='bold',
            labelpad=15,
            fontname='Times New Roman',
            color='black'
        )
        ax.set_ylabel(
            pretty_name(feature_2),
            fontsize=38,
            weight='bold',
            labelpad=15,
            fontname='Times New Roman',
            color='black'
        )

        if x_max - x_min > 0:
            ax.set_xticks(np.linspace(x_min, x_max, 6))
        if y_max - y_min > 0:
            ax.set_yticks(np.linspace(y_min, y_max, 6))

        set_axis_one_decimal(ax)
        style_axis(ax, tick_size=28, spine_width=3.0)

        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')

        plt.tight_layout()

        png_path = os.path.join(output_dir, f'{clean_f1}_vs_{clean_f2}.png')
        pdf_path = os.path.join(output_dir, f'{clean_f1}_vs_{clean_f2}.pdf')

        plt.savefig(png_path, format='png', bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        plt.close(fig)

        success_count += 1
        if (i + 1) % 5 == 0:
            print(f"  已生成 {i + 1}/{len(all_pairs)} 个图")

    except Exception as e:
        failed_pairs.append((feature_1, feature_2, str(e)))
        plt.close('all')
        print(f"✗ 生成 {pretty_name(feature_1)} vs {pretty_name(feature_2)} 失败: {e}")

print(f"\n双变量图生成完成: {success_count}/{len(all_pairs)} 成功")

# =========================
# 单特征PDP图
# =========================
print("\n生成单特征偏依赖图...")
single_success_count = 0

for feature in features:
    try:
        clean_feature = clean_filename(pretty_name(feature))
        index = features.get_loc(feature)

        pdp_display = PartialDependenceDisplay.from_estimator(
            gbr,
            X_scaled,
            [index],
            grid_resolution=20,
            random_state=42
        )

        x_values = pdp_display.pd_results[0].grid_values[0]
        y_values = pdp_display.pd_results[0].average

        x_original = x_values * scaler.scale_[index] + scaler.mean_[index]
        y_original = y_scaler.inverse_transform(y_values.reshape(-1, 1)).ravel()

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.plot(x_original, y_original, 'b-', linewidth=2.5)
        ax.scatter(x_original, y_original, color='red', s=30, alpha=0.6)

        ax.fill_between(
            x_original, 0, y_original,
            where=(y_original > 0),
            color='blue',
            alpha=0.2
        )
        ax.fill_between(
            x_original, 0, y_original,
            where=(y_original < 0),
            color='red',
            alpha=0.2
        )

        ax.set_xlabel(
            pretty_name(feature),
            fontsize=18,
            weight='bold',
            fontname='Times New Roman',
            labelpad=12,
            color='black'
        )
        ax.set_ylabel(
            'Partial Dependence',
            fontsize=18,
            weight='bold',
            fontname='Times New Roman',
            labelpad=12,
            color='black'
        )

        if len(x_original) > 0 and x_original.max() - x_original.min() > 0:
            ax.set_xticks(np.linspace(x_original.min(), x_original.max(), 5))

        if len(y_original) > 0 and y_original.max() - y_original.min() > 0:
            ax.set_yticks(np.linspace(y_original.min(), y_original.max(), 5))

        set_axis_one_decimal(ax)
        style_axis(ax, tick_size=16, spine_width=1.5)

        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)

        importance = feature_importance.loc[
            feature_importance['Feature'] == feature, 'Importance'
        ].values[0]

        ax.text(
            0.02, 0.98,
            f'Importance: {importance:.4f}',
            transform=ax.transAxes,
            fontsize=12,
            fontname='Times New Roman',
            verticalalignment='top',
            weight='bold',
            color='black',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, linewidth=1)
        )

        plt.tight_layout()

        png_path = os.path.join(output_dir, f'Single_PDP_{clean_feature}.png')
        pdf_path = os.path.join(output_dir, f'Single_PDP_{clean_feature}.pdf')

        plt.savefig(png_path, format='png', bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        plt.close(fig)

        single_success_count += 1
        print(f"  ✓ 已生成 {pretty_name(feature)} 的单特征图")

    except Exception as e:
        print(f"✗ 单特征图失败: {pretty_name(feature)}, 错误: {e}")

print(f"单特征图生成完成: {single_success_count}/{len(features)} 成功")

# =========================
# 交互热图
# =========================
print("\n生成交互热图...")

interaction_df = pd.DataFrame(interaction_matrix, index=features, columns=features)
interaction_range_df = pd.DataFrame(interaction_range_matrix, index=features, columns=features)
interaction_std_df = pd.DataFrame(interaction_std_matrix, index=features, columns=features)

interaction_df_plot = interaction_df.copy()
interaction_range_df_plot = interaction_range_df.copy()
interaction_std_df_plot = interaction_std_df.copy()

interaction_df_plot.index = [pretty_name(x) for x in interaction_df_plot.index]
interaction_df_plot.columns = [pretty_name(x) for x in interaction_df_plot.columns]

interaction_range_df_plot.index = [pretty_name(x) for x in interaction_range_df_plot.index]
interaction_range_df_plot.columns = [pretty_name(x) for x in interaction_range_df_plot.columns]

interaction_std_df_plot.index = [pretty_name(x) for x in interaction_std_df_plot.index]
interaction_std_df_plot.columns = [pretty_name(x) for x in interaction_std_df_plot.columns]

mask = np.zeros_like(interaction_matrix, dtype=bool)
np.fill_diagonal(mask, True)

# 平均值热图
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = sns.heatmap(
    interaction_df_plot,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    mask=mask,
    square=True,
    cbar_kws={"label": "Interaction strength (mean)", "shrink": 0.8},
    annot_kws={"size": 12, "weight": "bold", "family": "Times New Roman"},
    linewidths=1,
    linecolor='white',
    ax=ax
)
cbar = heatmap.collections[0].colorbar
cbar.ax.set_ylabel('Interaction strength (mean)', fontsize=14, weight='bold', fontname='Times New Roman')
set_colorbar_one_decimal(cbar)

plt.xticks(rotation=45, ha='right', fontsize=14, weight='bold', fontname='Times New Roman')
plt.yticks(fontsize=14, weight='bold', fontname='Times New Roman')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "关键因子交互热图_平均值.png"), format='png', bbox_inches='tight', facecolor='white', dpi=300)
plt.savefig(os.path.join(output_dir, "关键因子交互热图_平均值.pdf"), format='pdf', bbox_inches='tight', facecolor='white')
plt.close()

# 范围热图
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = sns.heatmap(
    interaction_range_df_plot,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    mask=mask,
    square=True,
    cbar_kws={"label": "Interaction strength (range)", "shrink": 0.8},
    annot_kws={"size": 12, "weight": "bold", "family": "Times New Roman"},
    linewidths=1,
    linecolor='white',
    ax=ax
)
cbar = heatmap.collections[0].colorbar
cbar.ax.set_ylabel('Interaction strength (range)', fontsize=14, weight='bold', fontname='Times New Roman')
set_colorbar_one_decimal(cbar)

plt.xticks(rotation=45, ha='right', fontsize=14, weight='bold', fontname='Times New Roman')
plt.yticks(fontsize=14, weight='bold', fontname='Times New Roman')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "关键因子交互热图_范围.png"), format='png', bbox_inches='tight', facecolor='white', dpi=300)
plt.savefig(os.path.join(output_dir, "关键因子交互热图_范围.pdf"), format='pdf', bbox_inches='tight', facecolor='white')
plt.close()

# 标准差热图
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = sns.heatmap(
    interaction_std_df_plot,
    annot=True,
    fmt=".1f",
    cmap="Greens",
    mask=mask,
    square=True,
    cbar_kws={"label": "Interaction strength (std)", "shrink": 0.8},
    annot_kws={"size": 12, "weight": "bold", "family": "Times New Roman"},
    linewidths=1,
    linecolor='white',
    ax=ax
)
cbar = heatmap.collections[0].colorbar
cbar.ax.set_ylabel('Interaction strength (std)', fontsize=14, weight='bold', fontname='Times New Roman')
set_colorbar_one_decimal(cbar)

plt.xticks(rotation=45, ha='right', fontsize=14, weight='bold', fontname='Times New Roman')
plt.yticks(fontsize=14, weight='bold', fontname='Times New Roman')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "关键因子交互热图_标准差.png"), format='png', bbox_inches='tight', facecolor='white', dpi=300)
plt.savefig(os.path.join(output_dir, "关键因子交互热图_标准差.pdf"), format='pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("✓ 交互热图已生成")

# =========================
# 保存数据和模型
# =========================
print("\n保存数据文件...")

interaction_df.to_csv(os.path.join(output_dir, "关键因子交互矩阵_平均值.csv"), encoding='utf-8-sig')
interaction_range_df.to_csv(os.path.join(output_dir, "关键因子交互矩阵_范围.csv"), encoding='utf-8-sig')
interaction_std_df.to_csv(os.path.join(output_dir, "关键因子交互矩阵_标准差.csv"), encoding='utf-8-sig')

model_path = os.path.join(output_dir, "gradient_boosting_model.pkl")
scaler_path = os.path.join(output_dir, "scaler.pkl")
y_scaler_path = os.path.join(output_dir, "y_scaler.pkl")

joblib.dump(gbr, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(y_scaler, y_scaler_path)

print("✓ 模型和预处理器已保存")

# =========================
# 总结报告
# =========================
print("\n生成总结报告...")

summary_report = f"""
{'=' * 60}
总氮关键因子交互作用分析报告
{'=' * 60}
分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据文件: {os.path.basename(data_file)}
目标变量: {target_col} (索引: {list(data.columns).index(target_col)})

{'=' * 60}
关键因子 ({len(available_features)}个)
{'=' * 60}
{chr(10).join([f'  {i + 1}. {pretty_name(feature)}' for i, feature in enumerate(available_features)])}

{'=' * 60}
数据统计
{'=' * 60}
• 原始数据形状: {data.shape}
• 关键因子数量: {len(available_features)}
• 样本数量: {X.shape[0]}
• 训练集大小: {X_train.shape[0]}
• 测试集大小: {X_test.shape[0]}

{'=' * 60}
模型性能
{'=' * 60}
• R² 分数: {r2:.6f}
• 均方误差 (MSE): {mse:.6f}
• 均方根误差 (RMSE): {np.sqrt(mse):.6f}

{'=' * 60}
特征重要性排序
{'=' * 60}
{chr(10).join([f'  {i + 1}. {pretty_name(row.Feature):30s}: {row.Importance:.6f}' for i, row in enumerate(feature_importance.itertuples())])}

{'=' * 60}
生成的图表
{'=' * 60}
• 双变量交互图: {success_count}/{len(all_pairs)} 个
• 单特征偏依赖图: {single_success_count}/{len(features)} 个
• 交互热图: 3个

{'=' * 60}
失败的对
{'=' * 60}
{f'{chr(10).join([f"  - {pretty_name(pair[0])} × {pretty_name(pair[1])}: {pair[2]}" for pair in failed_pairs])}' if failed_pairs else '  无'}

{'=' * 60}
输出目录
{'=' * 60}
{output_dir}
"""

report_path = os.path.join(output_dir, "分析总结报告.txt")
with open(report_path, "w", encoding='utf-8') as f:
    f.write(summary_report)

print("✓ 总结报告已保存")

# =========================
# 最终输出
# =========================
print("\n" + "=" * 60)
print("🎉 分析完成!")
print("=" * 60)
print(f"📁 输出目录: {output_dir}")

print("\n💡 关键发现:")
print(f"目标变量: {target_col}")
print(f"最重要的因子: {pretty_name(feature_importance.iloc[0]['Feature'])} (重要性: {feature_importance.iloc[0]['Importance']:.4f})")
print(f"模型解释力 (R²): {r2:.4f}")

if not interaction_df.empty:
    max_interaction = interaction_df.stack().nlargest(10)
    printed = 0
    print("\n最强的交互作用:")
    for idx, value in max_interaction.items():
        if idx[0] != idx[1]:
            print(f"  {pretty_name(idx[0])} × {pretty_name(idx[1])}: {value:.4f}")
            printed += 1
        if printed >= 3:
            break

print("\n✅ 所有分析已完成!")