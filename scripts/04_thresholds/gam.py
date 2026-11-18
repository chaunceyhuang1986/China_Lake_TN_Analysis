import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pygam import LinearGAM, s, f

warnings.filterwarnings("ignore")

# ============================================================
# 0. 路径与参数
# ============================================================

FILE_PATH = os.environ.get("TN_MERGED_INPUT", "data/private/merged_lake_data_complete.csv")
OUT_DIR = os.environ.get("TN_OUTPUT_DIR", "outputs/gam")

# 数据中的 Lake surface area 原始单位为 m²；模型和图中转换为 km²
LAKE_AREA_INPUT_UNIT = "m2"

# 湖泊分组变量：惩罚因子项作为湖泊随机截距的近似
GROUP_COLUMN = "Hylak_id"

TN_CANDIDATES = ["TN", "总氮", "total_nitrogen", "Nitrogen", "TN_mg/L", "TN_mgL"]

FEATURES = [
    "PLAND-Cropland",
    "PLAND-Grassland",
    "PLAND-Water",
    "PLAND-Impervious",
    "Lake surface area",
    "HFI"
]

ALIASES = {
    "Lake surface area": [
        "Lake surface area",
        "Lake surface area (m²)",
        "Lake surface area (m2)",
        "Lake surface area (km²)",
        "Lake surface area (km2)"
    ]
}

# ============================================================
# 1. 字体、颜色与图形参数（沿用原图风格）
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold"
})

BG = "#F8F9FA"
TEXT = "#212529"
GRID = "#F0F0F0"
CURVE = "#2E5984"
BAND = "#2E5984"
THRESHOLD = "#D32F2F"
BAND_ALPHA = 0.20

# GAM平滑复杂度。先控制曲线复杂度，避免把细小波动识别成多个转折点。
N_SPLINES = 6
SMOOTH_LAMBDA = 0.6
RANDOM_EFFECT_LAMBDA = 10.0

# 转折点识别参数
EDGE_FRACTION = 0.05
MIN_RUN_FRACTION = 0.015


def resolve_column(df, candidates):
    """忽略大小写匹配列名。"""
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for name in candidates:
        if name in df.columns:
            return name

        key = str(name).strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def smart_tick_formatter(x, pos=None):
    if not np.isfinite(x):
        return ""

    if np.isclose(x, round(x), atol=1e-8):
        return str(int(round(x)))

    return f"{x:.3f}".rstrip("0").rstrip(".")


def style_axis(ax):
    formatter = FuncFormatter(smart_tick_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for label in labels:
            label.set_fontname("Times New Roman")
            label.set_fontweight("bold")

    ax.xaxis.label.set_fontname("Times New Roman")
    ax.yaxis.label.set_fontname("Times New Roman")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")

    ax.tick_params(axis="both", pad=4)
    ax.grid(True, color=GRID, alpha=0.7, linewidth=0.8)


def detect_candidate_turning_points(x_grid, fitted_log_tn):
    """
    根据平滑曲线斜率持续变号识别候选转折点。

    注意：这是候选转折点筛选规则，不等同于已通过统计检验的阈值。
    """
    derivative = np.gradient(fitted_log_tn, x_grid)

    # 对导数做轻度移动平均，避免网格数值波动造成过多变号
    window = 9
    kernel = np.ones(window) / window
    derivative_smooth = np.convolve(derivative, kernel, mode="same")

    crossing_indices = np.where(
        np.diff(np.sign(derivative_smooth)) != 0
    )[0] + 1

    edge = max(1, int(len(x_grid) * EDGE_FRACTION))
    run = max(5, int(len(x_grid) * MIN_RUN_FRACTION))

    candidates = []

    for idx in crossing_indices:
        if idx < edge or idx >= len(x_grid) - edge:
            continue

        left = derivative_smooth[max(edge, idx - run):idx]
        right = derivative_smooth[idx:min(len(x_grid) - edge, idx + run)]

        if len(left) < 5 or len(right) < 5:
            continue

        left_sign = np.sign(np.median(left))
        right_sign = np.sign(np.median(right))

        # 要求转折点两侧的斜率方向不同且在局部范围内保持稳定
        left_stable = np.mean(np.sign(left) == left_sign) >= 0.8
        right_stable = np.mean(np.sign(right) == right_sign) >= 0.8

        if left_stable and right_stable and left_sign != right_sign:
            candidates.append(idx)

    return candidates


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # -------------------- 读取数据 --------------------
    try:
        df = pd.read_csv(FILE_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_PATH, encoding="gbk")

    target_col = resolve_column(df, TN_CANDIDATES)
    if target_col is None:
        raise ValueError(
            "找不到 TN 目标变量列。请检查 CSV 列名。"
            f"\n当前列名：{list(df.columns)}"
        )

    if GROUP_COLUMN not in df.columns:
        raise ValueError(
            f"找不到随机效应分组列 {GROUP_COLUMN}。"
            "请确认数据中存在湖泊或监测站分组字段。"
        )

    source_map = {}
    for feature in FEATURES:
        candidates = ALIASES.get(feature, [feature])
        source_col = resolve_column(df, candidates)

        if source_col is None:
            raise ValueError(f"找不到指标列：{feature}")

        source_map[feature] = source_col

    # -------------------- 数据整理 --------------------
    data = pd.DataFrame(index=df.index)

    for feature, source_col in source_map.items():
        data[feature] = pd.to_numeric(df[source_col], errors="coerce")

    data["TN"] = pd.to_numeric(df[target_col], errors="coerce")
    data[GROUP_COLUMN] = df[GROUP_COLUMN]

    data = data.replace([np.inf, -np.inf], np.nan)

    # 面积单位转换：m² -> km²
    if LAKE_AREA_INPUT_UNIT.lower() in ("m2", "m²"):
        data["Lake surface area"] = data["Lake surface area"] / 1_000_000.0
    elif LAKE_AREA_INPUT_UNIT.lower() not in ("km2", "km²"):
        raise ValueError("LAKE_AREA_INPUT_UNIT 只能为 m2 或 km2。")

    # -------------------- TN数据诊断 --------------------
    tn_diag = pd.DataFrame([{
        "target_column": target_col,
        "raw_rows": len(df),
        "TN_valid": int(data["TN"].notna().sum()),
        "TN_missing_or_non_numeric": int(data["TN"].isna().sum()),
        "TN_negative": int((data["TN"] < 0).sum()),
        "TN_zero": int((data["TN"] == 0).sum()),
        "TN_min": data["TN"].min(),
        "TN_median": data["TN"].median(),
        "TN_max": data["TN"].max(),
        "n_lakes": int(data[GROUP_COLUMN].nunique(dropna=True))
    }])

    tn_diag.to_csv(
        out / "TN_Data_Diagnostics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTN数据诊断：")
    print(tn_diag.to_string(index=False))

    if (data["TN"] < 0).any():
        raise ValueError(
            "TN列存在负值。请检查目标列映射、单位或原始数据；"
            "诊断表已输出到结果目录。"
        )

    # -------------------- 六个因子诊断 --------------------
    feature_diag = []

    for feature in FEATURES:
        values = data[feature]
        valid_pair = data[[feature, "TN"]].dropna()

        feature_diag.append({
            "factor": feature,
            "source_column": source_map[feature],
            "n_valid_with_TN": len(valid_pair),
            "n_missing_or_non_numeric": int(values.isna().sum()),
            "n_unique": int(values.nunique(dropna=True)),
            "minimum": values.min(),
            "median": values.median(),
            "maximum": values.max()
        })

    feature_diag_df = pd.DataFrame(feature_diag)
    feature_diag_df.to_csv(
        out / "Six_Factor_Data_Diagnostics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n六个因子数据诊断：")
    print(feature_diag_df.to_string(index=False))

    # 湖泊ID转成0到n-1的类别编码，供pyGAM的因子项使用
    data["Lake_RE"] = pd.Categorical(data[GROUP_COLUMN]).codes
    data.loc[data["Lake_RE"] < 0, "Lake_RE"] = np.nan

    # -------------------- 作图与建模 --------------------
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 10.0))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()

    threshold_records = []
    model_records = []

    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        ax.set_facecolor(BG)

        panel = f"({chr(97 + i)})"
        x_label = (
            "Lake surface area (km²)"
            if feature == "Lake surface area"
            else feature
        )

        ax.set_title(
            panel,
            loc="left",
            fontsize=20,
            fontweight="bold",
            fontname="Times New Roman",
            color=TEXT
        )
        ax.set_xlabel(x_label, fontweight="bold")
        ax.set_ylabel("Total nitrogen concentration", fontweight="bold")

        d = data[[feature, "TN", "Lake_RE"]].dropna().sort_values(feature)

        if len(d) < 30 or d[feature].nunique() < 6:
            ax.text(
                0.5, 0.5, "Insufficient data",
                transform=ax.transAxes,
                ha="center", va="center",
                color="gray",
                fontweight="bold",
                fontname="Times New Roman"
            )
            style_axis(ax)
            continue

        x = d[feature].to_numpy(dtype=float)
        y = d["TN"].to_numpy(dtype=float)
        group = d["Lake_RE"].to_numpy(dtype=int)

        if np.any(y < 0):
            raise ValueError(f"{feature}建模数据中发现负TN，请检查原始记录。")

        # log1p允许TN=0；反变换后预测值不会低于0
        y_model = np.log1p(y)

        unique_x = np.unique(x).size
        n_splines = min(N_SPLINES, max(4, unique_x - 1))

        # s(0)：因子的非线性平滑项
        # f(1)：湖泊ID的惩罚因子项，近似湖泊随机截距
        X_model = np.column_stack([x, group])

        gam = LinearGAM(
            s(0, n_splines=n_splines) + f(1),
            lam=[SMOOTH_LAMBDA, RANDOM_EFFECT_LAMBDA]
        ).fit(X_model, y_model)

        x_grid = np.linspace(np.min(x), np.max(x), 500)
        group_codes = np.sort(np.unique(group))

        # 对湖泊随机效应进行边际平均，避免只展示某一个湖泊的曲线
        pred_by_lake = []
        ci_low_by_lake = []
        ci_high_by_lake = []

        for lake_code in group_codes:
            X_new = np.column_stack([
                x_grid,
                np.full(len(x_grid), lake_code)
            ])

            pred_by_lake.append(gam.predict(X_new))

            ci = gam.confidence_intervals(X_new, width=0.95)
            ci_low_by_lake.append(ci[:, 0])
            ci_high_by_lake.append(ci[:, 1])

        pred_log = np.mean(pred_by_lake, axis=0)
        ci_low_log = np.mean(ci_low_by_lake, axis=0)
        ci_high_log = np.mean(ci_high_by_lake, axis=0)

        # 回到TN原始浓度尺度；log1p反变换保证预测不为负
        pred_tn = np.maximum(np.expm1(pred_log), 0)
        ci_low_tn = np.maximum(np.expm1(ci_low_log), 0)
        ci_high_tn = np.maximum(np.expm1(ci_high_log), 0)

        # 候选转折点按 log1p(TN) 平滑曲线导数变号识别
        candidate_indices = detect_candidate_turning_points(x_grid, pred_log)

        ax.fill_between(
            x_grid,
            ci_low_tn,
            ci_high_tn,
            color=BAND,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=1
        )

        ax.plot(
            x_grid,
            pred_tn,
            color=CURVE,
            linewidth=4.0,
            zorder=3
        )

        for idx in candidate_indices:
            x_turn = x_grid[idx]
            y_turn = pred_tn[idx]

            ax.axvline(
                x=x_turn,
                color=THRESHOLD,
                linestyle="--",
                linewidth=2.2,
                alpha=0.85,
                zorder=4
            )

            ax.annotate(
                f"Candidate: {x_turn:.3g}",
                (x_turn, y_turn),
                xytext=(5, 8),
                textcoords="offset points",
                color=THRESHOLD,
                fontsize=10,
                fontweight="bold",
                fontname="Times New Roman",
                rotation=90,
                va="bottom"
            )

            threshold_records.append({
                "Factor": feature,
                "Candidate_turning_point": x_turn,
                "Fitted_TN_at_point": y_turn,
                "N": len(d),
                "N_lakes": int(d["Lake_RE"].nunique()),
                "Interpretation": (
                    "探索性候选转折点；需结合导数置信区间和敏感性分析验证"
                )
            })

        if not candidate_indices:
            threshold_records.append({
                "Factor": feature,
                "Candidate_turning_point": np.nan,
                "Fitted_TN_at_point": np.nan,
                "N": len(d),
                "N_lakes": int(d["Lake_RE"].nunique()),
                "Interpretation": "未识别到持续斜率变号的候选转折点"
            })

        pseudo_r2 = gam.statistics_.get("pseudo_r2", {})
        model_records.append({
            "Factor": feature,
            "N": len(d),
            "N_lakes": int(d["Lake_RE"].nunique()),
            "Pseudo_R2_explained_deviance": pseudo_r2.get(
                "explained_deviance", np.nan
            ),
            "N_candidate_turning_points": len(candidate_indices)
        })

        style_axis(ax)

    # -------------------- 输出图和结果表 --------------------
    plt.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.10,
        top=0.97,
        wspace=0.28,
        hspace=0.32
    )

    fig.savefig(
        out / "TN_Six_Factors_GAMM.png",
        dpi=600,
        bbox_inches="tight",
        facecolor=BG
    )
    fig.savefig(
        out / "TN_Six_Factors_GAMM.pdf",
        bbox_inches="tight",
        facecolor=BG
    )
    plt.close(fig)

    pd.DataFrame(threshold_records).to_csv(
        out / "TN_Six_Factors_Candidate_Turning_Points.csv",
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(model_records).to_csv(
        out / "TN_Six_Factors_Model_Summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n完成。结果保存至：", out)
    print("红线为候选转折点，不自动等于已验证阈值。")
    print("TN已用log1p尺度拟合并反变换，预测值不会为负。")
    print("Hylak_id已作为惩罚因子项纳入，模拟湖泊随机截距。")


if __name__ == "__main__":
    main()