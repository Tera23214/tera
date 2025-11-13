#!/usr/bin/env python3
"""
自动化 Epoch 扫描测试脚本 - Step 1 (20k baseline)

用途：
    测试 Step1 优化版本（Adam + LR Scheduling）在 20k baseline 下的加速效果。
    通过更大的 epoch 数，验证优化在充分训练场景下的实际收益。

使用方法:
    cd optimization_tests && python test_epoch_sweep_step1_20k.py

配置说明:
    - BASELINE_EPOCHS: 20000 (充分训练的基准)
    - STEP1_EPOCHS: [2000, 4000, 8000, 12000, 16000, 20000]
    - TOLERANCE: 对拍允许的相对误差 10%

输出:
    - 每个 epoch 的对拍结果
    - 最小通过 epoch 和加速倍数
    - 在更大 epoch 下的优化效果评价
"""

import subprocess
import json
import re
import sys
import time
from pathlib import Path

# ============================================================
# 配置参数
# ============================================================
BASELINE_SCRIPT = 'baseline/program/Main_baseline_for_comparison.py'
STEP1_SCRIPT = 'step1_adam_scheduler/program/Main_step1_adam_scheduler.py'

BASELINE_EPOCHS = 20000
STEP1_EPOCHS = [2000, 4000, 8000, 12000, 16000, 20000]
TOLERANCE = 0.10

# Programs output to Result folder (relative to project root)
# When running from optimization_tests/, the path is Result/200_200_50
RESULT_DIR = Path('Result/200_200_50')

# ============================================================
# 辅助函数
# ============================================================

def modify_epochs_in_file(file_path, epochs):
    """修改文件中的 EPOCHS_PER_ALPHA 参数"""
    print(f"  修改 {file_path} 中的 EPOCHS_PER_ALPHA = {epochs}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换 EPOCHS_PER_ALPHA 行
    pattern = r'EPOCHS_PER_ALPHA = \d+'
    if not re.search(pattern, content):
        print(f"    ⚠️  警告: 未找到 EPOCHS_PER_ALPHA 参数")
        return False

    new_content = re.sub(pattern, f'EPOCHS_PER_ALPHA = {epochs}', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True


def run_training(script, label):
    """运行训练脚本"""
    print(f"\n{'='*80}")
    print(f"运行 {script} ({label})")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Use current Python interpreter to inherit conda environment
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time

        print(f"✅ 训练完成，耗时 {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ 训练失败，耗时 {elapsed:.1f}s")
        print(f"错误信息:\n{e.stderr}")
        return False


def find_latest_result_file(prefix='results'):
    """找到最新生成的结果文件"""
    if not RESULT_DIR.exists():
        print(f"    ⚠️  结果目录不存在: {RESULT_DIR}")
        return None

    result_files = sorted(
        RESULT_DIR.glob(f'{prefix}*.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not result_files:
        print(f"    ⚠️  未找到 {prefix}*.json 文件")
        return None

    print(f"    找到结果文件: {result_files[0].name}")
    return result_files[0]


def get_result_file(epochs, is_step1=False):
    """获取指定epoch的结果文件路径"""
    if is_step1:
        filename = f'results_step1_epoch{epochs}.json'
    else:
        filename = f'results_epoch{epochs}.json'

    filepath = RESULT_DIR / filename
    if filepath.exists():
        return filepath

    # 如果不存在,尝试找最新的文件
    prefix = 'results_step1' if is_step1 else 'results'
    latest = find_latest_result_file(prefix)
    if latest:
        # 重命名为期望的文件名
        latest.rename(filepath)
        print(f"    重命名为: {filename}")
        return filepath

    return None


def compare_results(baseline_file, new_file):
    """运行对拍并返回是否通过"""
    print(f"\n  对拍验证:")
    print(f"    Baseline: {baseline_file.name}")
    print(f"    New:      {new_file.name}")

    result = subprocess.run(
        [sys.executable, 'compare_with_phase_check.py', str(baseline_file), str(new_file), str(TOLERANCE)],
        capture_output=True,
        text=True
    )

    # 显示对拍输出
    print(result.stdout)

    return result.returncode == 0


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*80)
    print("Step 1 验证: Epoch 扫描测试 (Adam + LR Scheduling)")
    print("="*80)
    print(f"目标: 找到 Step1 能达到 Baseline 效果的最小 epoch")
    print(f"Baseline: {BASELINE_EPOCHS} epochs (固定)")
    print(f"Step1 测试: {STEP1_EPOCHS}")
    print(f"对拍容差: {TOLERANCE*100}%")
    print("="*80)

    # ============================================================
    # 阶段 1: 运行 Baseline
    # ============================================================
    print(f"\n{'='*80}")
    print(f"阶段 1: 运行 Baseline (固定 {BASELINE_EPOCHS} epochs)")
    print(f"{'='*80}")

    if not modify_epochs_in_file(BASELINE_SCRIPT, BASELINE_EPOCHS):
        print("❌ 修改 Baseline 配置失败")
        return 1

    if not run_training(BASELINE_SCRIPT, f'Baseline {BASELINE_EPOCHS} epochs'):
        print("❌ Baseline 训练失败")
        return 1

    baseline_file = get_result_file(BASELINE_EPOCHS, is_step1=False)
    if not baseline_file:
        print("❌ 未找到 Baseline 结果文件")
        return 1

    # ============================================================
    # 阶段 2: 测试 Step1 不同 epochs
    # ============================================================
    print(f"\n{'='*80}")
    print(f"阶段 2: 测试 Step1 不同 epochs")
    print(f"{'='*80}")

    results = {}

    for epochs in STEP1_EPOCHS:
        print(f"\n{'-'*80}")
        print(f"测试 Step1: {epochs} epochs")
        print(f"{'-'*80}")

        # 修改配置
        if not modify_epochs_in_file(STEP1_SCRIPT, epochs):
            print(f"  ⚠️  跳过: 修改配置失败")
            results[epochs] = None
            continue

        # 运行训练
        if not run_training(STEP1_SCRIPT, f'Step1 {epochs} epochs'):
            print(f"  ❌ 训练失败")
            results[epochs] = None
            continue

        # 获取结果文件
        step1_file = get_result_file(epochs, is_step1=True)
        if not step1_file:
            print(f"  ⚠️  未找到结果文件")
            results[epochs] = None
            continue

        # 对拍验证
        passed = compare_results(baseline_file, step1_file)
        results[epochs] = passed

        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n  结果: {epochs} epochs -> {status}")

    # ============================================================
    # 阶段 3: 汇总结果
    # ============================================================
    print(f"\n{'='*80}")
    print(f"阶段 3: 汇总结果")
    print(f"{'='*80}")
    print(f"\nBaseline: {BASELINE_EPOCHS} epochs (固定)\n")

    for epochs in STEP1_EPOCHS:
        result = results.get(epochs)
        if result is None:
            status = "⚠️  SKIP (训练失败)"
        elif result:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"  Step1 {epochs:5d} epochs: {status}")

    # 找到最小通过的 epoch
    passing_epochs = [e for e, p in results.items() if p is True]

    print(f"\n{'='*80}")
    print(f"最终评价")
    print(f"{'='*80}")

    if passing_epochs:
        min_passing = min(passing_epochs)
        speedup = BASELINE_EPOCHS / min_passing

        print(f"\n✅ 最小通过 epoch: {min_passing}")
        print(f"✅ 加速倍数: {speedup:.2f}x")

        # 评价
        if speedup >= 5.0:
            print(f"\n🎉🎉🎉 优秀！加速 {speedup:.2f}x，优化效果显著！")
            print(f"       继续进行后续优化步骤。")
            return_code = 0
        elif speedup >= 2.0:
            print(f"\n✅✅ 良好！加速 {speedup:.2f}x，优化有效。")
            print(f"       继续进行后续优化步骤，期待累积效果。")
            return_code = 0
        elif speedup >= 1.2:
            print(f"\n⚠️  勉强通过。加速 {speedup:.2f}x，效果有限。")
            print(f"   可以继续，但效果可能不如预期。")
            return_code = 0
        else:
            print(f"\n❌ 失败！加速 {speedup:.2f}x < 1.2x，优化效果不明显。")
            print(f"   建议检查代码实现或调整超参数。")
            return_code = 1
    else:
        print(f"\n❌ 失败: 即使 {max(STEP1_EPOCHS)} epochs 也未通过对拍。")
        print(f"   Step1 需要 ≥ Baseline epoch，这个优化没有意义。")
        print(f"   需要检查代码实现或超参数。")
        return_code = 1

    print(f"{'='*80}\n")
    return return_code


if __name__ == '__main__':
    sys.exit(main())
