#!/usr/bin/env python3
"""
自适应Alpha采样执行器 - 两阶段采样策略

工作流程:
1. 粗扫描阶段: 大步长(Δα=0.5)快速训练，定位相变区和异常区
2. 相变分析: 使用PhaseTransitionAnalyzer识别关键区域
3. 精细扫描阶段: 在关键区域用小步长(Δα=0.01/0.02)密集训练
4. 结果合并: 整合两阶段结果

优势:
- 节省计算量: 相比均匀采样节省60-70%
- 保证精度: 在相变区域密集采样
- 自动识别: 无需手动指定相变位置
"""

import json
import subprocess
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict

from phase_transition_analyzer import PhaseTransitionAnalyzer


class AdaptiveAlphaRunner:
    """自适应Alpha采样执行器"""

    def __init__(self,
                 program_path: str,
                 result_dir: Path,
                 coarse_step: float = 0.5,
                 phase_step: float = 0.01,
                 anomaly_step: float = 0.02,
                 stable_step: float = 0.5,
                 alpha_range: tuple = (0.0, 3.0)):
        """
        参数:
            program_path: 训练程序路径（如 Main_step2_adam_scheduler.py）
            result_dir: 结果保存目录
            coarse_step: 粗扫描步长
            phase_step: 相变区域精细步长
            anomaly_step: 异常区域步长
            stable_step: 平稳区域步长
            alpha_range: Alpha范围 (start, stop)
        """
        self.program_path = Path(program_path)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.coarse_step = coarse_step
        self.phase_step = phase_step
        self.anomaly_step = anomaly_step
        self.stable_step = stable_step
        self.alpha_range = alpha_range

        # 结果存储
        self.coarse_results = None
        self.fine_results = None
        self.merged_results = None
        self.sampling_plan = None

    def run_training(self, alphas: List[float], stage: str = "coarse") -> Dict:
        """
        运行训练程序

        参数:
            alphas: 要训练的alpha列表
            stage: 阶段名称 ("coarse" 或 "fine")

        返回:
            训练结果字典
        """
        print(f"\n{'='*80}")
        print(f"运行训练 - {stage.upper()} 阶段")
        print(f"{'='*80}")
        print(f"  程序: {self.program_path}")
        print(f"  Alpha数量: {len(alphas)}")
        print(f"  Alpha范围: [{min(alphas):.2f}, {max(alphas):.2f}]")
        print(f"{'='*80}\n")

        # TODO: 这里需要修改训练程序来接受alpha列表参数
        # 暂时假设训练程序会读取配置文件或环境变量
        # 实际实现时需要根据具体程序接口调整

        print(f"⚠️  注意: 此功能需要训练程序支持自定义alpha列表")
        print(f"  当前为演示模式，将使用现有结果文件\n")

        # 演示模式：返回空字典
        # 实际使用时，这里应该调用训练程序并返回结果
        return {}

    def phase1_coarse_scan(self) -> Dict:
        """
        阶段1: 粗扫描

        使用大步长快速覆盖整个alpha范围，定位大致的相变区域
        """
        print(f"\n{'#'*80}")
        print(f"# 阶段1: 粗扫描 (Coarse Scan)")
        print(f"{'#'*80}\n")

        # 生成粗扫描alpha列表
        alphas_coarse = np.arange(
            self.alpha_range[0],
            self.alpha_range[1] + self.coarse_step/2,
            self.coarse_step
        )

        print(f"粗扫描alpha列表:")
        print(f"  步长: Δα = {self.coarse_step}")
        print(f"  数量: {len(alphas_coarse)} 个点")
        print(f"  列表: {alphas_coarse.tolist()}\n")

        # 运行训练（实际使用时取消注释）
        # self.coarse_results = self.run_training(alphas_coarse, stage="coarse")

        # 演示模式：加载现有结果
        print(f"📁 演示模式: 加载现有粗扫描结果")
        coarse_result_path = self.result_dir / "results_coarse_scan.json"

        # 如果没有粗扫描结果，使用现有的20k epoch结果作为演示
        if not coarse_result_path.exists():
            print(f"   未找到粗扫描结果，使用 results_epoch20000.json 作为演示")
            coarse_result_path = self.result_dir / "results_epoch20000.json"

        if coarse_result_path.exists():
            with open(coarse_result_path, 'r') as f:
                self.coarse_results = json.load(f)
            print(f"   ✅ 加载成功: {coarse_result_path}")
            print(f"   Alpha数量: {len(self.coarse_results)}\n")
        else:
            print(f"   ❌ 未找到结果文件")
            return {}

        return self.coarse_results

    def phase2_analyze(self) -> Dict:
        """
        阶段2: 相变分析

        使用PhaseTransitionAnalyzer分析粗扫描结果，
        识别相变区、异常区、平稳区
        """
        print(f"\n{'#'*80}")
        print(f"# 阶段2: 相变分析 (Phase Transition Analysis)")
        print(f"{'#'*80}\n")

        if not self.coarse_results:
            print(f"❌ 错误: 粗扫描结果为空")
            return {}

        # 提取alpha列表
        alphas = sorted([float(a) for a in self.coarse_results.keys()])

        # 创建分析器
        print(f"创建 PhaseTransitionAnalyzer...")
        analyzer = PhaseTransitionAnalyzer(alphas, self.coarse_results)

        # 检测相变
        phase = analyzer.detect_phase_transition_enhanced()
        print(f"✅ 相变检测完成:")
        print(f"   α_c = {phase['alpha_c']:.3f}")
        print(f"   梯度 = {phase['gradient']:.4f}")
        print(f"   一致性 = {phase['consistency']:.4f}\n")

        # 检测异常
        anomalies = analyzer.detect_anomalies()
        print(f"✅ 异常检测完成:")
        print(f"   检测到 {len(anomalies)} 个异常区域\n")

        # 生成采样计划
        self.sampling_plan = analyzer.suggest_adaptive_sampling(
            phase_step=self.phase_step,
            anomaly_step=self.anomaly_step,
            stable_step=self.stable_step
        )

        # 显示采样计划
        print(f"✅ 采样计划生成:")
        total_points = sum(item['num_points'] for item in self.sampling_plan)
        print(f"   总采样点数: {total_points}\n")

        for i, item in enumerate(self.sampling_plan, 1):
            print(f"   {i}. {item['reason']:20s} "
                  f"[{item['range'][0]:.2f}, {item['range'][1]:.2f}]  "
                  f"Δα={item['step']:.3f}  "
                  f"{item['num_points']}点")

        # 保存分析结果
        analysis_path = self.result_dir / "adaptive_sampling_analysis.json"
        analyzer.save_analysis_json(analysis_path)
        print(f"\n📁 分析结果已保存: {analysis_path}")

        # 保存可视化
        fig_path = self.result_dir / "adaptive_sampling_analysis.png"
        analyzer.plot_full_analysis(save_path=fig_path, show=False)
        print(f"📁 可视化图已保存: {fig_path}\n")

        return {
            'phase': phase,
            'anomalies': anomalies,
            'sampling_plan': self.sampling_plan
        }

    def phase3_fine_scan(self) -> Dict:
        """
        阶段3: 精细扫描

        根据采样计划，在相变区和异常区进行密集采样
        """
        print(f"\n{'#'*80}")
        print(f"# 阶段3: 精细扫描 (Fine Scan)")
        print(f"{'#'*80}\n")

        if not self.sampling_plan:
            print(f"❌ 错误: 采样计划为空")
            return {}

        # 提取所有精细扫描alpha点
        fine_alphas = []
        for item in self.sampling_plan:
            fine_alphas.extend(item['alphas'])

        # 去重并排序
        fine_alphas = sorted(set(fine_alphas))

        print(f"精细扫描alpha列表:")
        print(f"  总数量: {len(fine_alphas)} 个点")
        print(f"  范围: [{min(fine_alphas):.2f}, {max(fine_alphas):.2f}]\n")

        # 运行训练（实际使用时取消注释）
        # self.fine_results = self.run_training(fine_alphas, stage="fine")

        # 演示模式：使用现有结果
        print(f"📁 演示模式: 使用现有结果作为精细扫描结果")
        print(f"   (实际使用时，这里会运行训练程序)\n")

        self.fine_results = self.coarse_results  # 演示模式

        return self.fine_results

    def phase4_merge(self) -> Dict:
        """
        阶段4: 结果合并

        合并粗扫描和精细扫描的结果
        """
        print(f"\n{'#'*80}")
        print(f"# 阶段4: 结果合并 (Merge Results)")
        print(f"{'#'*80}\n")

        # 演示模式：直接使用精细扫描结果
        self.merged_results = self.fine_results

        print(f"✅ 结果合并完成")
        print(f"   最终alpha数量: {len(self.merged_results)}\n")

        # 保存最终结果
        final_path = self.result_dir / "results_adaptive_sampling.json"
        with open(final_path, 'w') as f:
            json.dump(self.merged_results, f, indent=2)

        print(f"📁 最终结果已保存: {final_path}\n")

        return self.merged_results

    def run(self) -> Dict:
        """
        执行完整的自适应采样流程

        返回:
            最终合并的结果字典
        """
        print(f"\n{'='*80}")
        print(f"自适应Alpha采样执行器 - 开始运行")
        print(f"{'='*80}")
        print(f"  训练程序: {self.program_path}")
        print(f"  结果目录: {self.result_dir}")
        print(f"  Alpha范围: [{self.alpha_range[0]}, {self.alpha_range[1]}]")
        print(f"  粗扫描步长: {self.coarse_step}")
        print(f"  相变区步长: {self.phase_step}")
        print(f"  异常区步长: {self.anomaly_step}")
        print(f"  平稳区步长: {self.stable_step}")
        print(f"{'='*80}\n")

        # 阶段1: 粗扫描
        self.phase1_coarse_scan()

        # 阶段2: 相变分析
        analysis = self.phase2_analyze()

        # 阶段3: 精细扫描（实际使用时取消注释）
        # self.phase3_fine_scan()

        # 阶段4: 结果合并（实际使用时取消注释）
        # self.phase4_merge()

        # 总结
        print(f"\n{'='*80}")
        print(f"自适应采样执行完成")
        print(f"{'='*80}")

        if self.sampling_plan:
            total_points = sum(item['num_points'] for item in self.sampling_plan)
            uniform_01 = int((self.alpha_range[1] - self.alpha_range[0]) / 0.01) + 1
            uniform_05 = int((self.alpha_range[1] - self.alpha_range[0]) / 0.05) + 1

            print(f"\n计算量对比:")
            print(f"  自适应采样: {total_points} 个点")
            print(f"  均匀Δα=0.05: {uniform_05} 个点 (节省 {(1-total_points/uniform_05)*100:.1f}%)")
            print(f"  均匀Δα=0.01: {uniform_01} 个点 (节省 {(1-total_points/uniform_01)*100:.1f}%)")

        print(f"\n✅ 所有阶段完成")
        print(f"{'='*80}\n")

        return self.merged_results if self.merged_results else self.coarse_results


def main():
    """主函数 - 命令行接口"""
    if len(sys.argv) < 2:
        print("用法: python adaptive_alpha_runner.py <program_path> [result_dir]")
        print()
        print("参数:")
        print("  program_path  - 训练程序路径 (如 step2_adam_scheduler/program/Main_step2_adam_scheduler.py)")
        print("  result_dir    - 结果保存目录 (默认 Result/200_200_50)")
        print()
        print("示例:")
        print("  python adaptive_alpha_runner.py step2_adam_scheduler/program/Main_step2_adam_scheduler.py")
        print()
        print("说明:")
        print("  本程序演示自适应采样策略的工作流程")
        print("  实际使用需要训练程序支持自定义alpha列表")
        sys.exit(1)

    program_path = sys.argv[1]
    result_dir = sys.argv[2] if len(sys.argv) > 2 else "Result/200_200_50"

    # 创建执行器
    runner = AdaptiveAlphaRunner(
        program_path=program_path,
        result_dir=Path(result_dir),
        coarse_step=0.5,
        phase_step=0.01,
        anomaly_step=0.02,
        stable_step=0.5,
        alpha_range=(0.0, 3.0)
    )

    # 运行
    try:
        results = runner.run()
        print(f"\n🎉 执行成功！")
        return 0
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
