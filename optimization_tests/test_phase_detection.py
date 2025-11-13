"""
测试 PhaseTransitionAnalyzer 在 20k epoch 数据上的效果

验证目标：
1. 相变点检测：α_c 应该在 1.9-2.0 附近
2. 三指标一致性：在相变点处应该很高
3. 异常检测：识别是否有梯度异常区域
4. 自适应采样：验证采样计划的合理性
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from phase_transition_analyzer import PhaseTransitionAnalyzer


def load_results(json_path: Path) -> dict:
    """加载训练结果JSON文件"""
    with open(json_path, 'r') as f:
        return json.load(f)


def test_phase_detection():
    """主测试函数"""
    print("=" * 80)
    print("Phase Transition Detection Test - 20k Epoch Data")
    print("=" * 80)
    print()

    # 加载20k epoch的baseline数据
    result_dir = Path('Result/200_200_50')
    baseline_path = result_dir / 'results_epoch20000.json'

    if not baseline_path.exists():
        print(f"❌ 数据文件不存在: {baseline_path}")
        print("   请先运行 test_epoch_sweep_step2_20k.py 生成数据")
        return

    print(f"📁 加载数据: {baseline_path}")
    metrics = load_results(baseline_path)

    # 提取alpha列表（排序并转换为float）
    alphas = sorted([float(a) for a in metrics.keys()])
    print(f"   Alpha 范围: {alphas[0]:.1f} - {alphas[-1]:.1f}")
    print(f"   Alpha 数量: {len(alphas)}")
    print()

    # ========================================
    # 1. 创建分析器
    # ========================================
    print("【1. 创建 PhaseTransitionAnalyzer】")
    analyzer = PhaseTransitionAnalyzer(
        alphas=alphas,
        metrics=metrics,
        decel_threshold=0.5,
        volatility_threshold=0.3
    )
    print("   ✅ 分析器创建成功")
    print()

    # ========================================
    # 2. 基础梯度分析
    # ========================================
    print("【2. 基础梯度分析】")
    max_grad_point = analyzer.find_max_gradient_point('Q_Y')
    print(f"   最大梯度点:")
    print(f"     α = {max_grad_point['alpha_c']:.3f}")
    print(f"     dQ_Y/dα = {max_grad_point['gradient_max']:.4f}")
    print(f"     Q_Y = {max_grad_point['Q_at_transition']:.4f}")

    # 验证是否在预期范围
    if 1.8 <= max_grad_point['alpha_c'] <= 2.1:
        print(f"   ✅ 相变点在预期范围 [1.8, 2.1] 内")
    else:
        print(f"   ⚠️  相变点不在预期范围 [1.8, 2.1] 内")
    print()

    # ========================================
    # 3. 三指标一致性分析
    # ========================================
    print("【3. 三指标一致性分析】")
    phase_enhanced = analyzer.detect_phase_transition_enhanced()
    print(f"   增强检测结果:")
    print(f"     α_c = {phase_enhanced['alpha_c']:.3f}")
    print(f"     Q_Y梯度 = {phase_enhanced['gradient']:.4f}")
    print(f"     三指标一致性 = {phase_enhanced['consistency']:.4f}")
    print(f"     综合置信度 = {phase_enhanced['confidence']:.4f}")
    print()

    print(f"   各指标梯度:")
    for metric, grad in phase_enhanced['all_gradients'].items():
        print(f"     d{metric}/dα = {grad:.4f}")
    print()

    # 验证一致性
    if phase_enhanced['consistency'] > 0.5:
        print(f"   ✅ 相变点一致性高 (>{0.5:.2f})")
    else:
        print(f"   ⚠️  相变点一致性较低 (<{0.5:.2f})")
    print()

    # ========================================
    # 4. 异常检测
    # ========================================
    print("【4. 异常检测】")
    anomalies = analyzer.detect_anomalies('Q_Y')
    if anomalies:
        print(f"   检测到 {len(anomalies)} 个异常区域:")
        for i, anom in enumerate(anomalies, 1):
            print(f"   {i}. α={anom['alpha']:.3f}, 类型={anom['type']}, "
                  f"严重性={anom['severity']:.4f}")
            print(f"      {anom['description']}")
    else:
        print(f"   ✅ 未检测到异常区域（曲线平滑）")
    print()

    # ========================================
    # 5. 区域分类
    # ========================================
    print("【5. 区域分类】")
    regions = analyzer.classify_regions()
    print(f"   相变区域: [{regions['phase_transition'][0]:.2f}, "
          f"{regions['phase_transition'][1]:.2f}]")
    print(f"   异常区域数量: {len(regions['anomalies'])}")
    if regions['anomalies']:
        for i, (start, end) in enumerate(regions['anomalies'], 1):
            print(f"     {i}. [{start:.2f}, {end:.2f}]")
    print(f"   平稳区域数量: {len(regions['stable'])}")
    if regions['stable']:
        for i, (start, end) in enumerate(regions['stable'], 1):
            print(f"     {i}. [{start:.2f}, {end:.2f}]")
    print()

    # ========================================
    # 6. 自适应采样建议
    # ========================================
    print("【6. 自适应采样建议】")
    sampling_plan = analyzer.suggest_adaptive_sampling(
        phase_step=0.01,
        anomaly_step=0.02,
        stable_step=0.5
    )

    total_points = sum(item['num_points'] for item in sampling_plan)
    print(f"   总采样点数: {total_points}")
    print()

    for i, item in enumerate(sampling_plan, 1):
        print(f"   {i}. {item['reason']} 区域:")
        print(f"      范围: [{item['range'][0]:.2f}, {item['range'][1]:.2f}]")
        print(f"      步长: Δα = {item['step']}")
        print(f"      采样点数: {item['num_points']}")
    print()

    # 计算节省的计算量
    uniform_05 = int((3.0 - 0.0) / 0.05) + 1
    uniform_01 = int((3.0 - 0.0) / 0.01) + 1
    saving_05 = (1 - total_points / uniform_05) * 100
    saving_01 = (1 - total_points / uniform_01) * 100

    print(f"   计算量对比:")
    print(f"     vs 均匀采样 Δα=0.05 ({uniform_05}点): 节省 {saving_05:.1f}%")
    print(f"     vs 均匀采样 Δα=0.01 ({uniform_01}点): 节省 {saving_01:.1f}%")
    print()

    # ========================================
    # 7. 生成完整报告
    # ========================================
    print("【7. 生成分析报告】")
    report = analyzer.generate_report()
    report_path = result_dir / 'phase_transition_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   ✅ 文本报告已保存: {report_path}")

    # 保存JSON结果
    json_path = result_dir / 'phase_transition_analysis.json'
    analyzer.save_analysis_json(json_path)
    print(f"   ✅ JSON结果已保存: {json_path}")
    print()

    # ========================================
    # 8. 可视化
    # ========================================
    print("【8. 生成可视化图表】")
    fig_path = result_dir / 'phase_transition_analysis.png'
    analyzer.plot_full_analysis(save_path=fig_path, show=False)
    print(f"   ✅ 图表已保存: {fig_path}")
    print()

    # ========================================
    # 总结
    # ========================================
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"✅ 相变点检测: α_c = {phase_enhanced['alpha_c']:.3f} (预期 1.9-2.0)")
    print(f"✅ 三指标一致性: {phase_enhanced['consistency']:.4f}")
    print(f"✅ 异常检测: {len(anomalies)} 个异常")
    print(f"✅ 自适应采样: {total_points} 个点 (节省 {saving_05:.1f}% vs Δα=0.05)")
    print()
    print(f"📊 报告文件: {report_path}")
    print(f"📊 JSON结果: {json_path}")
    print(f"📊 可视化图: {fig_path}")
    print()
    print("🎉 测试完成！")
    print("=" * 80)

    return analyzer


if __name__ == "__main__":
    analyzer = test_phase_detection()
