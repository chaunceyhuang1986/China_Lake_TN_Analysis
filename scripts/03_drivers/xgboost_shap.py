import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
import shap
import warnings
from pathlib import Path

# =========================
# 全局字体设置
# =========================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 24
})

plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


# =========================
# 标签显示名称映射
# 只改图上的显示，不改真实列名
# =========================
DISPLAY_NAME_MAP = {
    # Precipitation regime
    'mean_pre': 'Pmean',
    'max_pre': 'Pmax',
    'min_pre': 'Pmin',

    # Air temperature regime
    'mean_tem': 'Tmean',
    'max_tem': 'Tmax',
    'min_tem': 'Tmin',

    'Lake surface area': 'Lake surface area',
    'HFI': 'HFI',
    'PLAND-Grassland': 'PLAND-Grassland',
    'PLAND-Cropland': 'PLAND-Cropland',
    'PLAND-Water': 'PLAND-Water',
    'PLAND-Forest': 'PLAND-Forest',
    'PLAND-Impervious': 'PLAND-Impervious'
}


def pretty_name(name):
    """
    只修改图上显示名称，不修改真实数据列名。
    """
    return DISPLAY_NAME_MAP.get(str(name).strip(), str(name))


def improved_data_preprocessing(df):
    """
    改进的数据预处理：
    1. 从特征中移除 Month；
    2. 保留 Month 用于季节划分；
    3. 模型仍然使用原始字段名。
    """
    print("=== 数据预处理（移除月份特征）===")

    if 'TN' not in df.columns:
        print("❌ 错误: 数据中不存在 'TN' 列")
        return None, None, None, None

    # =========================
    # 真实特征列名
    # 注意：这里不要改成全称
    # =========================
    base_features = [
        'mean_pre', 'max_pre', 'min_pre',
        'mean_tem', 'max_tem', 'min_tem',
        'Lake surface area', 'HFI',
        'PLAND-Grassland', 'PLAND-Cropland',
        'PLAND-Water', 'PLAND-Forest', 'PLAND-Impervious'
    ]

    exclude_features = ['Hylak_id', 'Lon', 'Lat', 'Year']
    exclude_features.append('Month')

    available_features = [
        col for col in base_features
        if col in df.columns and col not in exclude_features and col != 'TN'
    ]

    print(f"初始特征数量: {len(base_features)}")
    print("从特征中移除: Month (仅用于季节划分)")
    print(f"可用特征数量: {len(available_features)}")
    print(f"真实特征列表: {available_features}")
    print(f"图上显示标签: {[pretty_name(x) for x in available_features]}")

    X = df[available_features].copy()
    y = df['TN'].copy()

    if 'Month' in df.columns:
        month_data = df['Month'].copy()
    else:
        print("⚠️ 警告: 数据中没有 Month 列，无法进行季节分析")
        return None, None, None, None

    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    month_data = month_data[valid_idx]

    df_valid = df[valid_idx].copy()
    df_valid = df_valid[['TN', 'Month'] + available_features]

    print(f"有效样本数: {len(X)}")

    if len(X) == 0:
        return None, None, None, None

    # 转为数值型，避免字符串导致模型报错
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    numeric_imputer = KNNImputer(n_neighbors=5)
    X_imputed = numeric_imputer.fit_transform(X)

    X = pd.DataFrame(
        X_imputed,
        columns=X.columns,
        index=X.index
    )

    return X, y, available_features, df_valid


def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost only; missing xgboost is an installation error."""
    print("\n=== 训练 XGBoost 模型 ===")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"模型性能: R² = {r2:.4f}, RMSE = {rmse:.4f}")
    return model, X_train_scaled, X_test_scaled, r2, rmse, scaler

def create_seasonal_feature_importance_plot(
    model,
    X,
    feature_names,
    df,
    output_dir,
    overall_r2,
    overall_rmse
):
    """
    创建四季特征重要性合成图。
    注意：
    - feature_names 使用真实列名；
    - 图上的 y 轴标签使用 pretty_name() 显示全称。
    """
    print("\n=== 创建季节性特征重要性合成图 ===")

    if 'Month' not in df.columns:
        print("❌ 错误: 数据中缺少月份信息")
        return []

    seasons = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2]
    }

    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    seasonal_results = []

    for i, season_name in enumerate(season_order):
        print(f"处理 {season_name} 数据...")

        season_months = seasons[season_name]
        season_mask = df['Month'].isin(season_months).values

        X_season = X[season_mask]
        df_season = df[season_mask]

        if len(X_season) < 10:
            print(f"⚠️ {season_name} 样本不足 ({len(X_season)})，跳过")

            axes[i].text(
                0.5,
                0.5,
                f'{season_name}\nNo Data',
                ha='center',
                va='center',
                transform=axes[i].transAxes,
                fontsize=20,
                fontweight='bold',
                fontname='Times New Roman'
            )

            axes[i].set_title(
                f'{season_name}',
                fontsize=22,
                fontweight='bold',
                fontname='Times New Roman'
            )

            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            continue

        y_pred_season = model.predict(X_season)
        y_true_season = df_season['TN'].values

        r2_season = r2_score(y_true_season, y_pred_season)
        rmse_season = np.sqrt(mean_squared_error(y_true_season, y_pred_season))

        sample_size_season = min(300, len(X_season))
        sample_idx_season = np.random.choice(
            len(X_season),
            sample_size_season,
            replace=False
        )

        X_season_sample = X_season[sample_idx_season]

        explainer = shap.TreeExplainer(model)
        shap_values_season = explainer.shap_values(X_season_sample)

        mean_abs_shap = np.mean(np.abs(shap_values_season), axis=0)

        importance_df = pd.DataFrame({
            'Feature': feature_names,       # 真实列名
            'mean_SHAP': mean_abs_shap
        }).sort_values('mean_SHAP', ascending=True)

        importance_df = importance_df.tail(10)

        ax = axes[i]

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(importance_df)))

        bars = ax.barh(
            range(len(importance_df)),
            importance_df['mean_SHAP'],
            color=colors,
            height=0.6,
            alpha=0.8
        )

        ax.set_yticks(range(len(importance_df)))

        # =========================
        # 图上标签使用全称
        # 不改变 importance_df 中的真实列名
        # =========================
        y_labels = []

        for feature in importance_df['Feature']:
            label = pretty_name(feature)

            if len(label) > 30:
                y_labels.append(label[:28] + '...')
            else:
                y_labels.append(label)

        ax.set_yticklabels(
            y_labels,
            fontsize=22,
            fontname='Times New Roman',
            fontweight='bold'
        )

        ax.set_xlabel(
            'mean(|SHAP value|)',
            fontsize=22,
            fontweight='bold',
            fontname='Times New Roman'
        )

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=22
        )

        max_shap = importance_df['mean_SHAP'].max()
        x_max = max(0.25, max_shap * 1.2)
        ax.set_xlim(0, x_max)

        months_str = ', '.join(map(str, season_months))

        ax.set_title(
            f'{season_name} (Months: {months_str})',
            fontsize=28,
            fontweight='bold',
            pad=22,
            fontname='Times New Roman'
        )

        for label in ax.get_xticklabels():
            label.set_fontname('Times New Roman')

        for j, (bar, value) in enumerate(zip(bars, importance_df['mean_SHAP'])):
            if value > max_shap * 0.05:
                ax.text(
                    value + x_max * 0.01,
                    j,
                    f'{value:.3f}',
                    va='center',
                    fontsize=18,
                    fontweight='bold',
                    fontname='Times New Roman'
                )

        ax.grid(True, alpha=0.3, axis='x')

        seasonal_results.append({
            'Season': season_name,
            'Months': months_str,
            'Samples': len(X_season),
            'R2': r2_season,
            'RMSE': rmse_season,
            'Top_Feature': importance_df.iloc[-1]['Feature'],                 # 真实列名
            'Top_Feature_Display': pretty_name(importance_df.iloc[-1]['Feature']),  # 显示名
            'Top_Importance': importance_df.iloc[-1]['mean_SHAP']
        })

        print(
            f"✅ {season_name} 子图完成 "
            f"(R²: {r2_season:.3f}, 样本数: {len(X_season)})"
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    output_dir = Path(output_dir)

    plt.savefig(
        output_dir / "Seasonal_Feature_Importance_TimesNewRoman.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        output_dir / "Seasonal_Feature_Importance_TimesNewRoman.pdf",
        bbox_inches='tight'
    )

    plt.close()

    print("✅ 季节性特征重要性合成图已保存")

    return seasonal_results


def create_individual_seasonal_plots(
    model,
    X,
    feature_names,
    df,
    output_dir,
    overall_r2,
    overall_rmse
):
    """
    创建四个单独的季节图。
    注意：
    - feature_names 使用真实列名；
    - 图上 y 轴标签使用 pretty_name() 显示全称。
    """
    print("\n=== 创建单独的季节图 ===")

    seasons = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2]
    }

    seasonal_results = []

    for season_name, season_months in seasons.items():
        print(f"创建 {season_name} 单独图...")

        season_mask = df['Month'].isin(season_months).values

        X_season = X[season_mask]
        df_season = df[season_mask]

        if len(X_season) < 10:
            print(f"⚠️ {season_name} 样本不足，跳过")
            continue

        y_pred_season = model.predict(X_season)
        y_true_season = df_season['TN'].values

        r2_season = r2_score(y_true_season, y_pred_season)
        rmse_season = np.sqrt(mean_squared_error(y_true_season, y_pred_season))

        sample_size_season = min(500, len(X_season))
        sample_idx_season = np.random.choice(
            len(X_season),
            sample_size_season,
            replace=False
        )

        X_season_sample = X_season[sample_idx_season]

        explainer = shap.TreeExplainer(model)
        shap_values_season = explainer.shap_values(X_season_sample)

        plt.figure(figsize=(14, 10))

        mean_abs_shap = np.mean(np.abs(shap_values_season), axis=0)

        importance_df = pd.DataFrame({
            'Feature': feature_names,      # 真实列名
            'mean_SHAP': mean_abs_shap
        }).sort_values('mean_SHAP', ascending=True)

        importance_df = importance_df.tail(12)

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(importance_df)))

        bars = plt.barh(
            range(len(importance_df)),
            importance_df['mean_SHAP'],
            color=colors,
            height=0.6,
            alpha=0.8
        )

        # =========================
        # 图上标签使用全称
        # 不改变真实列名
        # =========================
        display_features = [
            pretty_name(feature)
            for feature in importance_df['Feature']
        ]

        plt.yticks(
            range(len(importance_df)),
            display_features,
            fontsize=24,
            fontname='Times New Roman',
            fontweight='bold'
        )

        plt.xlabel(
            'mean(|SHAP value|)',
            fontsize=29,
            fontweight='bold',
            fontname='Times New Roman'
        )

        plt.xticks(fontname='Times New Roman')
        plt.yticks(fontname='Times New Roman')

        max_shap = importance_df['mean_SHAP'].max()
        x_max = max(0.25, max_shap * 1.2)
        plt.xlim(0, x_max)

        months_str = ', '.join(map(str, season_months))

        plt.title(
            f'{season_name} (Months: {months_str}) - Feature Importance',
            fontsize=28,
            fontweight='bold',
            pad=24,
            fontname='Times New Roman'
        )

        for i, (bar, value) in enumerate(zip(bars, importance_df['mean_SHAP'])):
            if value > max_shap * 0.05:
                plt.text(
                    value + x_max * 0.01,
                    i,
                    f'{value:.3f}',
                    va='center',
                    fontsize=24,
                    fontweight='bold',
                    fontname='Times New Roman'
                )

        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        output_dir = Path(output_dir)

        filename = f"Individual_Feature_Importance_{season_name}_TimesNewRoman.png"

        plt.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches='tight'
        )

        plt.savefig(
            output_dir / filename.replace('.png', '.pdf'),
            bbox_inches='tight'
        )

        plt.close()

        seasonal_results.append({
            'Season': season_name,
            'Months': months_str,
            'Samples': len(X_season),
            'R2': r2_season,
            'RMSE': rmse_season,
            'Top_Feature': importance_df.iloc[-1]['Feature'],                 # 真实列名
            'Top_Feature_Display': pretty_name(importance_df.iloc[-1]['Feature']),  # 显示名
            'Top_Importance': importance_df.iloc[-1]['mean_SHAP']
        })

        print(f"✅ {season_name} 单独图已保存")

    return seasonal_results


def main_seasonal_analysis():
    """
    主函数。
    """
    input_file = os.environ.get("TN_MERGED_INPUT", "data/private/merged_lake_data_complete.csv")
    output_dir = os.environ.get("TN_OUTPUT_DIR", "outputs/xgboost_shap")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")

    try:
        # =========================
        # 1. 加载数据
        # =========================
        print("正在加载数据...")

        df = pd.read_csv(input_file)

        print(f"✅ 原始数据形状: {df.shape}")

        required_columns = ['TN', 'Month']
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            print(f"❌ 错误: 数据中缺少必要的列: {missing_columns}")
            return

        print(f"数据列: {df.columns.tolist()}")
        print(f"月份分布: {df['Month'].unique()}")

        seasons = {
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Autumn': [9, 10, 11],
            'Winter': [12, 1, 2]
        }

        print("\n季节分布统计:")

        for season_name, months in seasons.items():
            season_count = len(df[df['Month'].isin(months)])
            season_tn_mean = (
                df.loc[df['Month'].isin(months), 'TN'].mean()
                if season_count > 0
                else None
            )

            if season_tn_mean is not None:
                print(
                    f"  {season_name} (月份: {months}): "
                    f"{season_count} 个样本, TN 平均值: {season_tn_mean:.4f}"
                )
            else:
                print(
                    f"  {season_name} (月份: {months}): "
                    f"{season_count} 个样本"
                )

        # =========================
        # 2. 数据预处理
        # =========================
        X, y, feature_names, df_valid = improved_data_preprocessing(df)

        if X is None:
            return

        print("\n✅ 预处理完成")
        print(f"特征数量: {len(feature_names)}")
        print(f"真实特征列表: {feature_names}")
        print(f"显示标签列表: {[pretty_name(x) for x in feature_names]}")
        print(f"是否包含 Month: {'Month' in feature_names}")
        print(f"df_valid 中是否包含 Month: {'Month' in df_valid.columns}")

        # =========================
        # 3. 划分训练集和测试集
        # =========================
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        train_idx = X_train.index
        test_idx = X_test.index

        df_train = df_valid.loc[train_idx]
        df_test = df_valid.loc[test_idx]

        print(
            f"训练集: {X_train.shape}, "
            f"月份样本分布: {df_train['Month'].value_counts().sort_index().to_dict()}"
        )

        print(
            f"测试集: {X_test.shape}, "
            f"月份样本分布: {df_test['Month'].value_counts().sort_index().to_dict()}"
        )

        # =========================
        # 4. 训练模型
        # =========================
        model, X_train_scaled, X_test_scaled, r2, rmse, scaler = train_model(
            X_train,
            y_train,
            X_test,
            y_test
        )

        print(f"\n训练集形状: {X_train_scaled.shape}")
        print(f"df_train 形状: {df_train.shape}")
        print(f"df_train 中是否包含 Month: {'Month' in df_train.columns}")

        # =========================
        # 5. 创建季节性合成图
        # =========================
        seasonal_results = create_seasonal_feature_importance_plot(
            model,
            X_train_scaled,
            feature_names,
            df_train,
            output_dir,
            r2,
            rmse
        )

        # =========================
        # 6. 创建单独季节图
        # =========================
        individual_results = create_individual_seasonal_plots(
            model,
            X_train_scaled,
            feature_names,
            df_train,
            output_dir,
            r2,
            rmse
        )

        # =========================
        # 7. 保存季节性结果
        # =========================
        results_df = pd.DataFrame(seasonal_results)

        results_df.to_csv(
            Path(output_dir) / "季节性特征重要性汇总.csv",
            index=False,
            encoding='utf-8-sig'
        )

        # =========================
        # 8. 保存模型特征重要性
        # =========================
        feature_importance = pd.DataFrame({
            'Feature': feature_names,                         # 真实列名
            'Feature_Display': [pretty_name(x) for x in feature_names],  # 显示名
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        feature_importance.to_csv(
            Path(output_dir) / "模型特征重要性.csv",
            index=False,
            encoding='utf-8-sig'
        )

        print("\n✅ 季节性分析完成")
        print(f"整体模型性能: R² = {r2:.3f}, RMSE = {rmse:.3f}")
        print("主要输出文件:")
        print("   - Seasonal_Feature_Importance_TimesNewRoman.png")
        print("   - Seasonal_Feature_Importance_TimesNewRoman.pdf")
        print("   - 4 个单独的季节图")
        print("   - 季节性特征重要性汇总.csv")
        print("   - 模型特征重要性.csv")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_seasonal_analysis()
