"""
相变前斜率分析工具

分析不同(N, M)配置下Q_Y在相变前线性增长区域的斜率，
并拟合斜率与N/M的关系。

使用方法：
    1. 修改下方参数配置
    2. python analyze_slope_vs_ratio.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional

# ============================================================
# 参数配置 - 在这里修改
# ============================================================

# JSON结果文件路径（相对路径）
JSON_PATH = "Result_compareNM/200x50_400x50_800x50_2000x50_5000x50/multi_size_results_steps5000.json"

# 线性拟合的alpha范围
# 设为 None 则自动检测，设为具体值则手动指定
ALPHA_START = 0.5  # 例如: 0.5
ALPHA_END = 1.4    # 例如: 1.5

# 输出目录（None = 与输入文件同目录）
OUTPUT_DIR = None

# ============================================================


def load_results(json_path: str) -> dict:
    """加载JSON结果文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def parse_config_key(key: str) -> Tuple[int, int]:
    """解析配置键，如 '200x50' -> (200, 50)"""
    parts = key.split('x')
    return int(parts[0]), int(parts[1])


def extract_qy_data(results: dict, config_key: str, alpha_values: List[float]) -> np.ndarray:
    """提取指定配置的Q_Y数据"""
    config_results = results[config_key]
    qy = []
    for alpha in alpha_values:
        alpha_str = str(float(alpha))
        if alpha_str in config_results:
            qy.append(config_results[alpha_str]['Q_Y_mean'])
        else:
            # 尝试其他格式
            for k in config_results.keys():
                if abs(float(k) - alpha) < 1e-6:
                    qy.append(config_results[k]['Q_Y_mean'])
                    break
    return np.array(qy)


def detect_transition_point(alpha_values: np.ndarray, qy_values: np.ndarray) -> Tuple[float, dict]:
    """
    智能检测相变点（梯度峰值法）

    方法：
    1. 计算Q_Y的一阶导数（梯度）
    2. 找梯度最大的位置 = 相变点
    3. 同时返回相变区域的特征

    返回：(相变点alpha, 详细信息dict)
    """
    # 平滑处理（简单移动平均）
    window = 3
    if len(qy_values) > window * 2:
        qy_smooth = np.convolve(qy_values, np.ones(window)/window, mode='valid')
        alpha_smooth = alpha_values[window//2:-(window//2)]
    else:
        qy_smooth = qy_values
        alpha_smooth = alpha_values

    # 计算梯度
    gradient = np.gradient(qy_smooth, alpha_smooth)

    # 找梯度最大点 = 相变点
    max_grad_idx = np.argmax(gradient)
    transition_alpha = alpha_smooth[max_grad_idx]
    max_gradient = gradient[max_grad_idx]

    # 计算相变宽度（梯度超过最大值一半的区域）
    half_max = max_gradient / 2
    above_half = gradient > half_max
    if np.any(above_half):
        first_idx = np.argmax(above_half)
        last_idx = len(above_half) - 1 - np.argmax(above_half[::-1])
        transition_width = alpha_smooth[last_idx] - alpha_smooth[first_idx]
        transition_start = alpha_smooth[first_idx]
        transition_end = alpha_smooth[last_idx]
    else:
        transition_width = 0.2
        transition_start = transition_alpha - 0.1
        transition_end = transition_alpha + 0.1

    info = {
        'transition_alpha': transition_alpha,
        'max_gradient': max_gradient,
        'transition_width': transition_width,
        'transition_start': transition_start,
        'transition_end': transition_end,
        'gradient_curve': (alpha_smooth, gradient)
    }

    return transition_alpha, info


def detect_linear_region(alpha_values: np.ndarray, qy_values: np.ndarray,
                         min_alpha: float = 0.3) -> Tuple[float, float, dict]:
    """
    自动检测线性增长区域

    策略：
    1. 用梯度峰值法找相变点
    2. 起点：Q_Y > 0.02 且 alpha >= min_alpha 的第一个点
    3. 终点：相变开始点前0.1个alpha单位

    返回：(alpha_start, alpha_end, phase_info)
    """
    # 智能检测相变点
    transition_alpha, phase_info = detect_transition_point(alpha_values, qy_values)

    # 找起点：Q_Y首次明显大于0
    start_idx = 0
    for i, qy in enumerate(qy_values):
        if qy > 0.02 and alpha_values[i] >= min_alpha:
            start_idx = i
            break

    # 终点设为相变开始点前0.1个alpha单位
    end_alpha = phase_info['transition_start'] - 0.1

    # 找对应的索引
    end_idx = len(alpha_values) - 1
    for i, alpha in enumerate(alpha_values):
        if alpha > end_alpha:
            end_idx = max(start_idx + 2, i - 1)  # 至少保证3个点
            break

    return alpha_values[start_idx], alpha_values[end_idx], phase_info


def fit_linear_region(alpha_values: np.ndarray, qy_values: np.ndarray,
                      alpha_start: float, alpha_end: float) -> Tuple[float, float, float, float]:
    """
    对指定区间进行线性拟合

    返回：(斜率, 截距, R², 斜率标准误差)
    """
    # 筛选指定区间的数据
    mask = (alpha_values >= alpha_start) & (alpha_values <= alpha_end)
    x = alpha_values[mask]
    y = qy_values[mask]

    if len(x) < 3:
        print(f"  警告：数据点太少 ({len(x)}个)，无法可靠拟合")
        return 0.0, 0.0, 0.0, 0.0

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return slope, intercept, r_value**2, std_err


def fit_slope_vs_ratio(ratios_NM: np.ndarray, slopes: np.ndarray) -> Dict:
    """
    拟合斜率与M/N的关系

    输入的ratios_NM是N/M，函数内部转换为M/N进行分析

    主要模型：
    1. 过原点线性: slope = a * (M/N)  【物理上最合理】
    2. 带截距线性: slope = a * (M/N) + b
    3. 幂律: slope = a * (M/N)^b
    """
    results = {}

    # 转换为 M/N
    ratios = 1.0 / ratios_NM  # M/N

    # 1. 过原点线性拟合: slope = a * (M/N) 【最重要】
    # 使用最小二乘: a = sum(x*y) / sum(x^2)
    try:
        a_origin = np.sum(ratios * slopes) / np.sum(ratios ** 2)
        y_pred = a_origin * ratios
        ss_res = np.sum((slopes - y_pred) ** 2)
        ss_tot = np.sum(slopes ** 2)  # 过原点时，R²的计算方式不同
        r2_origin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['linear_origin'] = {
            'a': a_origin, 'R2': r2_origin,
            'formula': f'slope = {a_origin:.6f} * (M/N)'
        }
    except:
        results['linear_origin'] = None

    # 2. 带截距线性拟合: slope = a * (M/N) + b
    try:
        slope_lin, intercept_lin, r_lin, _, _ = stats.linregress(ratios, slopes)
        results['linear'] = {
            'a': slope_lin, 'b': intercept_lin, 'R2': r_lin**2,
            'formula': f'slope = {slope_lin:.6f} * (M/N) + {intercept_lin:.6f}'
        }
    except:
        results['linear'] = None

    # 3. 幂律拟合: slope = a * (M/N)^b
    try:
        def power_law(x, a, b):
            return a * np.power(x, b)
        popt, _ = curve_fit(power_law, ratios, slopes, p0=[1.0, 1.0], maxfev=5000)
        y_pred = power_law(ratios, *popt)
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum((slopes - np.mean(slopes))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['power_law'] = {
            'a': popt[0], 'b': popt[1], 'R2': r2,
            'formula': f'slope = {popt[0]:.6f} * (M/N)^{popt[1]:.4f}'
        }
    except Exception as e:
        results['power_law'] = None

    return results


def plot_analysis(alpha_values: np.ndarray, all_qy: Dict[str, np.ndarray],
                  linear_fits: Dict[str, Dict], slope_analysis: Dict,
                  save_dir: Path):
    """生成分析图表"""

    # 颜色调色板
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#a65628', '#f781bf', '#999999']

    # =========================================================
    # 图1: Q_Y曲线 + 线性拟合
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    for i, (config_key, qy) in enumerate(all_qy.items()):
        N, M = parse_config_key(config_key)
        ratio = N / M
        color = colors[i % len(colors)]

        # 原始数据
        ax1.plot(alpha_values, qy, 'o-', color=color, markersize=4,
                 linewidth=1.5, label=f'N={N}, M={M} (N/M={ratio:.0f})')

        # 线性拟合线
        if config_key in linear_fits:
            fit = linear_fits[config_key]
            x_fit = np.linspace(fit['alpha_start'], fit['alpha_end'], 50)
            y_fit = fit['slope'] * x_fit + fit['intercept']
            ax1.plot(x_fit, y_fit, '--', color=color, linewidth=2, alpha=0.7)

    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax1.set_ylabel(r'$Q_Y$', fontsize=14)
    ax1.set_title('Q_Y vs Alpha (with Linear Fits in Pre-Transition Region)', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    save_path1 = save_dir / 'qy_with_linear_fits.png'
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"图1已保存: {save_path1}")
    plt.close(fig1)

    # =========================================================
    # 图2: 斜率 vs M/N
    # =========================================================
    ratios_MN = []  # M/N
    slopes_list = []
    slope_errs = []
    labels = []

    for config_key, fit in linear_fits.items():
        N, M = parse_config_key(config_key)
        ratios_MN.append(M / N)
        slopes_list.append(fit['slope'])
        slope_errs.append(fit['slope_err'])
        labels.append(f'N={N}')

    ratios_MN = np.array(ratios_MN)
    slopes_arr = np.array(slopes_list)
    slope_errs = np.array(slope_errs)

    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # 数据点
    ax2.errorbar(ratios_MN, slopes_arr, yerr=slope_errs, fmt='o', markersize=10,
                 capsize=5, color='#2563eb', label='Measured slopes')

    # 添加标签
    for r, s, lbl in zip(ratios_MN, slopes_arr, labels):
        ax2.annotate(lbl, (r, s), textcoords="offset points", xytext=(5, 5), fontsize=9)

    # 拟合曲线
    if slope_analysis:
        x_fit = np.linspace(0, max(ratios_MN) * 1.1, 100)

        # 绘制所有拟合
        fit_colors = {'linear_origin': '#dc2626', 'linear': '#f59e0b', 'power_law': '#16a34a'}

        for model_name, result in slope_analysis.items():
            if result is None:
                continue

            if model_name == 'linear_origin':
                y_fit = result['a'] * x_fit
            elif model_name == 'linear':
                y_fit = result['a'] * x_fit + result['b']
            elif model_name == 'power_law':
                y_fit = result['a'] * np.power(np.maximum(x_fit, 1e-10), result['b'])
            else:
                continue  # 跳过未知模型

            # 过原点线性用实线加粗
            if model_name == 'linear_origin':
                linestyle = '-'
                linewidth = 2.5
                alpha = 1.0
            else:
                linestyle = '--'
                linewidth = 1.5
                alpha = 0.7

            label = f"{model_name}: R²={result['R2']:.4f}"
            ax2.plot(x_fit, y_fit, linestyle, color=fit_colors.get(model_name, '#999999'),
                     linewidth=linewidth, alpha=alpha, label=label)

    ax2.set_xlabel('M/N', fontsize=14)
    ax2.set_ylabel('Slope (dQ_Y/dα)', fontsize=14)
    ax2.set_title('Pre-Transition Slope vs M/N Ratio', fontsize=14, fontweight='bold')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    save_path2 = save_dir / 'slope_vs_ratio.png'
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"图2已保存: {save_path2}")
    plt.close(fig2)

    # =========================================================
    # 图3: 斜率 vs M/N (对数坐标)
    # =========================================================
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    ax3.loglog(ratios_MN, slopes_arr, 'o', markersize=10, color='#2563eb', label='Measured slopes')

    # 幂律拟合线
    if slope_analysis.get('power_law'):
        result = slope_analysis['power_law']
        x_fit = np.linspace(min(ratios_MN) * 0.9, max(ratios_MN) * 1.1, 100)
        y_fit = result['a'] * np.power(x_fit, result['b'])
        ax3.loglog(x_fit, y_fit, '-', color='#16a34a', linewidth=2,
                   label=f"Power law: slope ∝ (M/N)^{result['b']:.3f}, R²={result['R2']:.4f}")

    # 过原点线性
    if slope_analysis.get('linear_origin'):
        result = slope_analysis['linear_origin']
        x_fit = np.linspace(min(ratios_MN) * 0.9, max(ratios_MN) * 1.1, 100)
        y_fit = result['a'] * x_fit
        ax3.loglog(x_fit, y_fit, '--', color='#dc2626', linewidth=2,
                   label=f"Linear (origin): slope = {result['a']:.4f} * (M/N), R²={result['R2']:.4f}")

    ax3.set_xlabel('M/N (log scale)', fontsize=14)
    ax3.set_ylabel('Slope (log scale)', fontsize=14)
    ax3.set_title('Pre-Transition Slope vs M/N (Log-Log Plot)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    save_path3 = save_dir / 'slope_vs_ratio_loglog.png'
    fig3.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"图3已保存: {save_path3}")
    plt.close(fig3)


def generate_report(configs, linear_fits, slope_analysis, best_model) -> str:
    """生成自然语言分析报告"""
    lines = []
    lines.append("# 相变前斜率分析报告\n")

    # 数据概述
    lines.append("## 数据概述\n")
    lines.append(f"本次分析包含 **{len(configs)}** 组不同的 (N, M) 配置：\n")
    for config_key, N, M, ratio in configs:
        fit = linear_fits[config_key]
        lines.append(f"- N={N}, M={M} (N/M={ratio:.1f}): 斜率 = {fit['slope']:.4f}, 相变点 α_c ≈ {fit['transition_alpha']:.2f}")
    lines.append("")

    # 核心发现
    lines.append("## 核心发现\n")

    # 分析过原点线性拟合
    if slope_analysis.get('linear_origin'):
        lo = slope_analysis['linear_origin']
        lines.append(f"### 过原点线性拟合 (最重要)\n")
        lines.append(f"**公式**: `slope = {lo['a']:.4f} × (M/N)`\n")
        lines.append(f"**R² = {lo['R2']:.4f}**\n")

        if lo['R2'] > 0.95:
            lines.append("拟合效果优秀，数据强烈支持 **斜率与 M/N 成正比** 的假设。")
        elif lo['R2'] > 0.85:
            lines.append("拟合效果良好，数据基本支持斜率与 M/N 成正比的假设。")
        else:
            lines.append("拟合效果一般，可能存在其他影响因素。")
        lines.append("")

    # 分析幂律拟合
    if slope_analysis.get('power_law'):
        pl = slope_analysis['power_law']
        lines.append(f"### 幂律拟合\n")
        lines.append(f"**公式**: `slope = {pl['a']:.4f} × (M/N)^{pl['b']:.3f}`\n")
        lines.append(f"**R² = {pl['R2']:.4f}**\n")

        # 分析指数
        exp = pl['b']
        if abs(exp - 1.0) < 0.1:
            lines.append(f"幂律指数 **{exp:.3f} ≈ 1**，与线性关系一致，验证了 slope ∝ M/N 的假设。")
        elif exp > 1.0:
            lines.append(f"幂律指数 {exp:.3f} > 1，斜率随 M/N 增长略快于线性。")
        else:
            lines.append(f"幂律指数 {exp:.3f} < 1，斜率随 M/N 增长略慢于线性。可能由于有限尺寸效应或数据噪声。")
        lines.append("")

    # 物理解释
    lines.append("## 物理意义\n")
    lines.append("相变前线性增长区域的斜率反映了系统在低观测密度下的学习效率：\n")
    lines.append("- **斜率 ∝ M/N** 意味着：当隐变量维度 M 相对于观测维度 N 越大时，")
    lines.append("  每增加一个单位的观测密度 α，Q_Y 增长越快")
    lines.append("- 这与直觉一致：M/N 越大，信息冗余度越高，更容易从部分观测中恢复信号")
    lines.append("")

    # 结论
    lines.append("## 结论\n")

    if slope_analysis.get('linear_origin') and slope_analysis['linear_origin']['R2'] > 0.9:
        a = slope_analysis['linear_origin']['a']
        lines.append(f"数据支持以下经验公式：\n")
        lines.append(f"```")
        lines.append(f"dQ_Y/dα ≈ {a:.2f} × (M/N)    (相变前线性区)")
        lines.append(f"```\n")
        lines.append("即相变前 Q_Y 的增长速率与 M/N 成正比，比例系数约为 " + f"{a:.2f}。")
    else:
        lines.append(f"最佳拟合模型为 **{best_model}**，但数据可能需要更多样本点来确认关系。")

    return "\n".join(lines)


def main():
    # 使用顶部配置的参数
    json_path = Path(JSON_PATH)

    if not json_path.exists():
        print(f"错误：文件不存在 {json_path}")
        print(f"请检查 JSON_PATH 配置是否正确")
        return

    print("=" * 70)
    print("相变前斜率分析")
    print("=" * 70)
    print(f"输入文件: {json_path}")

    data = load_results(json_path)

    # 提取配置
    alpha_values = np.array(data['alpha_values'])
    results = data['results']
    config_keys = list(results.keys())

    print(f"Alpha范围: {alpha_values[0]:.2f} - {alpha_values[-1]:.2f} ({len(alpha_values)}个点)")
    print(f"配置数量: {len(config_keys)}")

    # 解析所有配置
    configs = []
    for key in config_keys:
        N, M = parse_config_key(key)
        configs.append((key, N, M, N/M))
        print(f"  - {key}: N={N}, M={M}, N/M={N/M:.1f}")

    # 按N/M排序
    configs.sort(key=lambda x: x[3])

    print("=" * 70)
    print("线性拟合分析")
    print("=" * 70)

    # 存储所有Q_Y数据和拟合结果
    all_qy = {}
    linear_fits = {}

    for config_key, N, M, ratio in configs:
        print(f"\n[N={N}, M={M}, N/M={ratio:.1f}]")

        # 提取Q_Y数据
        qy = extract_qy_data(results, config_key, alpha_values)
        all_qy[config_key] = qy

        # 检测线性区域和相变点
        if ALPHA_START is not None and ALPHA_END is not None:
            alpha_start, alpha_end = ALPHA_START, ALPHA_END
            _, phase_info = detect_transition_point(alpha_values, qy)
            print(f"  使用指定线性区域: [{alpha_start:.2f}, {alpha_end:.2f}]")
        else:
            alpha_start, alpha_end, phase_info = detect_linear_region(alpha_values, qy)
            print(f"  自动检测线性区域: [{alpha_start:.2f}, {alpha_end:.2f}]")

        # 输出相变点信息
        print(f"  检测到相变点: α_c = {phase_info['transition_alpha']:.3f}")
        print(f"  相变区域: [{phase_info['transition_start']:.2f}, {phase_info['transition_end']:.2f}]")
        print(f"  最大梯度: {phase_info['max_gradient']:.4f}")

        # 线性拟合
        slope, intercept, r2, slope_err = fit_linear_region(
            alpha_values, qy, alpha_start, alpha_end
        )

        print(f"  斜率: {slope:.6f} ± {slope_err:.6f}")
        print(f"  截距: {intercept:.6f}")
        print(f"  R²: {r2:.6f}")

        linear_fits[config_key] = {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'slope_err': slope_err,
            'alpha_start': alpha_start,
            'alpha_end': alpha_end,
            'transition_alpha': phase_info['transition_alpha'],
            'transition_start': phase_info['transition_start'],
            'transition_end': phase_info['transition_end'],
            'max_gradient': phase_info['max_gradient']
        }

    # 分析斜率与N/M的关系
    print("\n" + "=" * 70)
    print("斜率 vs N/M 关系拟合")
    print("=" * 70)

    ratios = np.array([c[3] for c in configs])
    slopes = np.array([linear_fits[c[0]]['slope'] for c in configs])

    slope_analysis = fit_slope_vs_ratio(ratios, slopes)

    for model_name, result in slope_analysis.items():
        if result is None:
            print(f"\n{model_name}: 拟合失败")
        else:
            print(f"\n{model_name}:")
            print(f"  公式: {result['formula']}")
            print(f"  R²: {result['R2']:.6f}")

    # 找最佳模型
    best_model = None
    best_r2 = -1
    for model_name, result in slope_analysis.items():
        if result and result['R2'] > best_r2:
            best_r2 = result['R2']
            best_model = model_name

    print(f"\n最佳模型: {best_model} (R² = {best_r2:.6f})")

    # 输出目录 - 放在子文件夹 slope_analysis/ 中
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = json_path.parent / 'slope_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    print("\n" + "=" * 70)
    print("生成图表")
    print("=" * 70)

    plot_analysis(alpha_values, all_qy, linear_fits, slope_analysis, output_dir)

    # 保存分析结果
    # 移除gradient_curve（不可JSON序列化）
    linear_fits_clean = {}
    for k, v in linear_fits.items():
        linear_fits_clean[k] = {kk: vv for kk, vv in v.items() if kk != 'gradient_curve'}

    analysis_results = {
        'input_file': str(json_path),
        'configs': [{'key': c[0], 'N': c[1], 'M': c[2], 'ratio': c[3]} for c in configs],
        'linear_fits': linear_fits_clean,
        'slope_vs_ratio': {k: v for k, v in slope_analysis.items() if v is not None},
        'best_model': best_model
    }

    analysis_path = output_dir / 'slope_analysis_results.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n分析结果已保存: {analysis_path}")

    # 生成自然语言报告
    report = generate_report(configs, linear_fits, slope_analysis, best_model)
    report_path = output_dir / 'analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分析报告已保存: {report_path}")

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
