import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import trange

def compute_auc01(y_true, y_pred):

    return roc_auc_score(y_true, y_pred, max_fpr=0.1)

def bootstrap_auc01_comparison(y, pA, pB, n_boot=1000, seed=42):
 
    np.random.seed(seed)
    n = len(y)

    # 在原始数据上计算 AUC0.1
    auc01_A = compute_auc01(y, pA)
    auc01_B = compute_auc01(y, pB)
    observed_diff = auc01_B - auc01_A

    print(f"Model A - Overall AUC0.1: {auc01_A:.4f}")
    print(f"Model B - Overall AUC0.1: {auc01_B:.4f}")
    print(f"Observed difference (B - A): {observed_diff:.4f}")

    # 存储每次 bootstrap 的差值
    diffs = []

    for _ in trange(n_boot, desc="Bootstrap Iterations"):
        # 有放回地采样
        indices = np.random.randint(0, n, n)
        y_boot = y[indices]
        pA_boot = pA[indices]
        pB_boot = pB[indices]

        auc01_A_boot = compute_auc01(y_boot, pA_boot)
        auc01_B_boot = compute_auc01(y_boot, pB_boot)
        diffs.append(auc01_B_boot - auc01_A_boot)

    diffs = np.array(diffs)
    mean_diff = diffs.mean()
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)

    # 单侧检验: B 是否显著大于 A
    # 统计在重采样分布中差值 <= 0 的比例
    p_value = np.sum(diffs <= 0) / n_boot

    return mean_diff, ci_low, ci_high, p_value

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_two_models_bootstrap.py <m1_file> <m2_file>")
        sys.exit(1)

    m1_file = sys.argv[1]
    m2_file = sys.argv[2]

    # 读取两个文件
    m1 = pd.read_csv(m1_file, sep="\t")
    m2 = pd.read_csv(m2_file, sep="\t")

    # 确保两个文件行数相同、且 'binder' 顺序一致
    if len(m1) != len(m2):
        print("Error: The two input files have different numbers of rows!")
        sys.exit(1)

    # 检查 binder 列是否完全相同
    if not np.all(m1['binder'].values == m2['binder'].values):
        print("Error: The 'binder' column differs between the two files!")
        sys.exit(1)

    # binder 作为真实标签
    y = m1['binder'].values
    # 分别获取两个文件的预测值
    pA = m1['prediction'].values  # Model A
    pB = m2['prediction'].values  # Model B

    # 进行 bootstrap 比较
    mean_diff, ci_low, ci_high, p_value = bootstrap_auc01_comparison(y, pA, pB, n_boot=2000, seed=42)

    print("\n=== Bootstrap Results ===")
    print(f"Mean difference in AUC0.1 (B - A): {mean_diff:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"One-sided p-value (B > A): {p_value:.8f}")

if __name__ == "__main__":
    main()

