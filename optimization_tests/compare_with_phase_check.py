#!/usr/bin/env python3
"""
增强对拍验证脚本 - 加入相变一致性检查

在原有对拍基础上，增加：
1. 相变点位置对比（α_c）
2. 相变锐度对比（最大梯度）
3. 三指标一致性对比
4. 异常区域对比

使用方法:
    python compare_with_phase_check.py baseline.json new.json [tolerance]
"""

import json
import sys
import numpy as np
from pathlib import Path

from phase_transition_analyzer import PhaseTransitionAnalyzer


def find_closest_alpha(keys, target, max_diff=0.01):
    """找到最接近目标 alpha 的 key"""
    min_diff = float('inf')
    closest = None
    for key in keys:
        try:
            val = float(key)
            diff = abs(val - target)
            if diff < min_diff:
                min_diff = diff
                closest = key
        except:
            continue
    return closest if min_diff < max_diff else None


def compare_in_safe_region(baseline_data, new_data,
                          safe_alphas=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                          tolerance=0.10):
    """
    在整个 alpha 范围 (0-3) 进行对拍

    20k epoch 下相变已经明显，需要验证整个范围包括相变区域。
    """

    print(f"\n{'='*80}")
    print(f"【1. 传统对拍验证】- 完整 Alpha 范围 (α ∈ [0, 3])")
    print(f"{'='*80}")
    print(f"测试点: {safe_alphas}")
    print(f"允许相对误差: {tolerance*100:.1f}%")
    print(f"{'='*80}\n")

    metrics = ['Q_Y_mean', 'Q_W_mean', 'Q_X_mean', 'Q_W_prime_mean', 'Q_X_prime_mean']
    metric_tolerance = {
        'Q_Y_mean': tolerance,
        'Q_W_mean': tolerance,
        'Q_X_mean': tolerance,
        'Q_W_prime_mean': tolerance * 1.5,  # Q' 允许稍大误差
        'Q_X_prime_mean': tolerance * 1.5,
    }

    passed = True
    all_errors = {m: [] for m in metrics}

    for alpha_val in safe_alphas:
        # 查找最接近的 alpha key
        baseline_key = find_closest_alpha(baseline_data.keys(), alpha_val)
        new_key = find_closest_alpha(new_data.keys(), alpha_val)

        if not baseline_key or not new_key:
            print(f"⚠️  α={alpha_val:.2f} 未找到对应数据")
            continue

        print(f"\n{'─'*80}")
        print(f"α = {alpha_val:.2f}")
        print(f"{'─'*80}")

        for metric in metrics:
            if metric not in baseline_data[baseline_key] or metric not in new_data[new_key]:
                continue

            base_val = baseline_data[baseline_key][metric]
            new_val = new_data[new_key][metric]
            rel_err = abs(new_val - base_val) / (abs(base_val) + 1e-6)
            all_errors[metric].append(rel_err)

            metric_tol = metric_tolerance[metric]
            status = "✅" if rel_err <= metric_tol else "❌"

            print(f"{status} {metric:16s}: "
                  f"baseline={base_val:.6f}  new={new_val:.6f}  "
                  f"err={rel_err*100:5.2f}% (tol={metric_tol*100:.1f}%)")

            if rel_err > metric_tol:
                passed = False

    # 统计摘要
    print(f"\n{'='*80}")
    print(f"对拍统计摘要")
    print(f"{'='*80}")

    for metric in metrics:
        if all_errors[metric]:
            errors = np.array(all_errors[metric])
            print(f"{metric:16s}: "
                  f"mean={np.mean(errors)*100:5.2f}%  "
                  f"max={np.max(errors)*100:5.2f}%  "
                  f"std={np.std(errors)*100:5.2f}%")

    print(f"{'='*80}")
    if passed:
        print(f"✅ 传统对拍通过")
    else:
        print(f"❌ 传统对拍失败")
    print(f"{'='*80}\n")

    return passed


def compare_phase_transitions(baseline_data, new_data,
                              alpha_c_tolerance=0.2,
                              gradient_tolerance=0.2,
                              consistency_tolerance=0.1):
    """
    相变一致性对比 - 核心增强功能

    对比两个版本的：
    1. 相变点位置 α_c（误差应 < 0.2）
    2. 相变锐度（最大梯度，误差应 < 20%）
    3. 三指标一致性（误差应 < 0.1）

    参数:
        alpha_c_tolerance: α_c 的绝对误差容忍度
        gradient_tolerance: 梯度的相对误差容忍度
        consistency_tolerance: 一致性的绝对误差容忍度
    """

    print(f"\n{'='*80}")
    print(f"【2. 相变一致性对比】- PhaseTransitionAnalyzer")
    print(f"{'='*80}\n")

    # 提取alpha列表
    baseline_alphas = sorted([float(a) for a in baseline_data.keys()])
    new_alphas = sorted([float(a) for a in new_data.keys()])

    # 创建分析器
    print("正在分析 Baseline 版本...")
    analyzer_baseline = PhaseTransitionAnalyzer(baseline_alphas, baseline_data)
    phase_baseline = analyzer_baseline.detect_phase_transition_enhanced()

    print("正在分析 New 版本...")
    analyzer_new = PhaseTransitionAnalyzer(new_alphas, new_data)
    phase_new = analyzer_new.detect_phase_transition_enhanced()

    print()
    print(f"{'─'*80}")
    print(f"相变检测结果对比")
    print(f"{'─'*80}")

    # 1. 相变点位置对比
    alpha_c_diff = abs(phase_baseline['alpha_c'] - phase_new['alpha_c'])
    alpha_c_ok = alpha_c_diff <= alpha_c_tolerance

    print(f"\n1. 相变点位置 (α_c):")
    print(f"   Baseline: α_c = {phase_baseline['alpha_c']:.3f}")
    print(f"   New:      α_c = {phase_new['alpha_c']:.3f}")
    print(f"   差异:     Δα_c = {alpha_c_diff:.3f}")
    print(f"   状态:     {'✅ 通过' if alpha_c_ok else '❌ 失败'} "
          f"(容忍度 < {alpha_c_tolerance})")

    # 2. 相变锐度对比（最大梯度）
    grad_baseline = phase_baseline['gradient']
    grad_new = phase_new['gradient']
    grad_rel_err = abs(grad_new - grad_baseline) / (abs(grad_baseline) + 1e-6)
    grad_ok = grad_rel_err <= gradient_tolerance

    print(f"\n2. 相变锐度 (最大梯度):")
    print(f"   Baseline: dQ_Y/dα = {grad_baseline:.4f}")
    print(f"   New:      dQ_Y/dα = {grad_new:.4f}")
    print(f"   差异:     {grad_rel_err*100:.2f}%")
    print(f"   状态:     {'✅ 通过' if grad_ok else '❌ 失败'} "
          f"(容忍度 < {gradient_tolerance*100:.1f}%)")

    # 3. 三指标一致性对比
    cons_baseline = phase_baseline['consistency']
    cons_new = phase_new['consistency']
    cons_diff = abs(cons_new - cons_baseline)
    cons_ok = cons_diff <= consistency_tolerance

    print(f"\n3. 三指标一致性:")
    print(f"   Baseline: {cons_baseline:.4f}")
    print(f"   New:      {cons_new:.4f}")
    print(f"   差异:     {cons_diff:.4f}")
    print(f"   状态:     {'✅ 通过' if cons_ok else '❌ 失败'} "
          f"(容忍度 < {consistency_tolerance})")

    # 4. 各指标梯度详细对比
    print(f"\n4. 各指标梯度详细对比:")
    grad_metrics = ['Q_Y', 'Q_W_prime', 'Q_X_prime']
    all_grad_ok = True

    for metric in grad_metrics:
        grad_b = phase_baseline['all_gradients'][metric]
        grad_n = phase_new['all_gradients'][metric]
        grad_err = abs(grad_n - grad_b) / (abs(grad_b) + 1e-6)
        grad_metric_ok = grad_err <= gradient_tolerance

        if not grad_metric_ok:
            all_grad_ok = False

        status = "✅" if grad_metric_ok else "❌"
        print(f"   {status} d{metric}/dα: "
              f"baseline={grad_b:.4f}, new={grad_n:.4f}, "
              f"err={grad_err*100:.2f}%")

    # 5. 异常区域对比
    print(f"\n5. 异常区域检测:")
    anomalies_baseline = analyzer_baseline.detect_anomalies()
    anomalies_new = analyzer_new.detect_anomalies()

    print(f"   Baseline: {len(anomalies_baseline)} 个异常")
    print(f"   New:      {len(anomalies_new)} 个异常")

    # 不要求异常区域完全一致，只做记录
    if anomalies_baseline:
        print(f"\n   Baseline 异常区域:")
        for i, anom in enumerate(anomalies_baseline[:5], 1):  # 最多显示5个
            print(f"     {i}. α={anom['alpha']:.3f}, 类型={anom['type']}")

    if anomalies_new:
        print(f"\n   New 异常区域:")
        for i, anom in enumerate(anomalies_new[:5], 1):
            print(f"     {i}. α={anom['alpha']:.3f}, 类型={anom['type']}")

    print(f"\n{'─'*80}")

    # 总体结论
    phase_check_ok = alpha_c_ok and grad_ok and cons_ok and all_grad_ok

    print(f"\n{'='*80}")
    print(f"相变一致性检查结果")
    print(f"{'='*80}")
    print(f"  α_c 位置一致:   {'✅ 通过' if alpha_c_ok else '❌ 失败'}")
    print(f"  锐度一致:       {'✅ 通过' if grad_ok else '❌ 失败'}")
    print(f"  三指标一致性:   {'✅ 通过' if cons_ok else '❌ 失败'}")
    print(f"  各指标梯度:     {'✅ 通过' if all_grad_ok else '❌ 失败'}")
    print(f"{'='*80}")

    if phase_check_ok:
        print(f"\n✅ 相变一致性检查通过！")
        print(f"   两个版本的相变特征高度一致，优化算法保持了物理行为。\n")
    else:
        print(f"\n❌ 相变一致性检查失败！")
        print(f"   两个版本的相变特征有显著差异，可能改变了学习动力学。\n")

    print(f"{'='*80}\n")

    return phase_check_ok, {
        'baseline': phase_baseline,
        'new': phase_new,
        'alpha_c_diff': alpha_c_diff,
        'gradient_rel_err': grad_rel_err,
        'consistency_diff': cons_diff
    }


def check_phase_transition_trend(results_data, name="新版本"):
    """
    检查相变区域的趋势合理性

    不要求精确对拍，只检查:
    1. Q_Y 单调递增
    2. 最终 Q_Y 接近 1
    3. 没有异常值 (NaN, Inf)
    """

    print(f"\n{'='*80}")
    print(f"【3. 趋势检查】- {name}")
    print(f"{'='*80}")

    alphas = sorted([float(k) for k in results_data.keys()])
    qy_values = [results_data[str(float(a))]['Q_Y_mean'] for a in alphas]

    # 显示所有点（简化版）
    print(f"\n  显示部分关键点:")
    key_indices = [0, len(alphas)//4, len(alphas)//2, 3*len(alphas)//4, -1]
    for i in key_indices:
        a, qy = alphas[i], qy_values[i]
        marker = "🔵" if a <= 1.5 else "🔴"
        print(f"  {marker} α={a:4.2f}  Q_Y={qy:.6f}")

    # 检查单调性
    non_monotonic_pairs = []
    for i in range(len(qy_values)-1):
        if qy_values[i] > qy_values[i+1] + 0.01:  # 允许 1% 的浮动
            non_monotonic_pairs.append((alphas[i], alphas[i+1], qy_values[i], qy_values[i+1]))

    is_monotonic = len(non_monotonic_pairs) == 0
    reaches_high = qy_values[-1] > 0.8
    no_nan = all(not np.isnan(qy) and not np.isinf(qy) for qy in qy_values)

    print(f"\n检查结果:")
    print(f"  单调递增: {'✅' if is_monotonic else '❌'}")
    if not is_monotonic:
        for a1, a2, q1, q2 in non_monotonic_pairs[:3]:  # 最多显示3个
            print(f"    α={a1:.2f} (Q_Y={q1:.4f}) > α={a2:.2f} (Q_Y={q2:.4f})")

    print(f"  最终 Q_Y > 0.8: {'✅' if reaches_high else '❌'} (Q_Y={qy_values[-1]:.6f})")
    print(f"  无异常值: {'✅' if no_nan else '❌'}")

    trend_ok = is_monotonic and reaches_high and no_nan
    print(f"\n{'✅ 趋势检查通过' if trend_ok else '❌ 趋势检查失败'}")
    print(f"{'='*80}\n")

    return trend_ok


def load_results(path):
    """加载结果文件"""
    path = Path(path)

    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    return data


def main():
    if len(sys.argv) < 3:
        print("用法: python compare_with_phase_check.py <baseline.json> <new.json> [tolerance]")
        print()
        print("参数:")
        print("  baseline.json  - 基线版本结果文件")
        print("  new.json       - 新版本结果文件")
        print("  tolerance      - 允许的相对误差 (默认 0.10 = 10%)")
        print()
        print("功能:")
        print("  1. 传统对拍验证（逐点对比）")
        print("  2. 相变一致性对比（α_c, 梯度, 一致性）")
        print("  3. 趋势检查（单调性）")
        print()
        print("示例:")
        print("  python compare_with_phase_check.py Result/baseline/results.json Result/step1/results.json 0.10")
        sys.exit(1)

    baseline_path = sys.argv[1]
    new_path = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

    print(f"\n{'='*80}")
    print(f"增强对拍验证 - 含相变一致性检查")
    print(f"{'='*80}")
    print(f"  Baseline: {baseline_path}")
    print(f"  New:      {new_path}")
    print(f"  Tolerance: {tolerance*100:.1f}%")
    print(f"{'='*80}\n")

    print(f"\n加载结果文件...")
    baseline_data = load_results(baseline_path)
    new_data = load_results(new_path)
    print(f"  Baseline: {len(baseline_data)} 个 alpha")
    print(f"  New:      {len(new_data)} 个 alpha")

    # ========================================
    # 1. 传统对拍（逐点对比）
    # ========================================
    passed_traditional = compare_in_safe_region(baseline_data, new_data, tolerance=tolerance)

    # ========================================
    # 2. 相变一致性对比（核心增强）
    # ========================================
    passed_phase, phase_details = compare_phase_transitions(
        baseline_data, new_data,
        alpha_c_tolerance=0.2,
        gradient_tolerance=0.2,
        consistency_tolerance=0.1
    )

    # ========================================
    # 3. 趋势检查
    # ========================================
    trend_ok = check_phase_transition_trend(new_data, name="New 版本")

    # ========================================
    # 4. 最终结论
    # ========================================
    print(f"\n{'='*80}")
    print(f"最终验证结果汇总")
    print(f"{'='*80}")
    print(f"  传统对拍:       {'✅ 通过' if passed_traditional else '❌ 失败'}")
    print(f"  相变一致性:     {'✅ 通过' if passed_phase else '❌ 失败'}")
    print(f"  趋势检查:       {'✅ 通过' if trend_ok else '❌ 失败'}")
    print(f"{'='*80}")

    all_passed = passed_traditional and passed_phase and trend_ok

    if all_passed:
        print(f"\n🎉 完整验证通过！")
        print(f"\n   ✅ 传统指标对拍通过")
        print(f"   ✅ 相变特征高度一致")
        print(f"   ✅ 学习曲线趋势合理")
        print(f"\n   结论: 新版本算法正确，优化有效，且保持了原有的物理行为。\n")
        return 0
    else:
        print(f"\n❌ 验证失败！")
        if not passed_traditional:
            print(f"   ❌ 传统对拍失败 - 部分指标超出容差")
        if not passed_phase:
            print(f"   ❌ 相变一致性失败 - 相变特征有显著差异")
        if not trend_ok:
            print(f"   ❌ 趋势检查失败 - 学习曲线异常")
        print(f"\n   需要检查算法实现或调整优化参数。\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
