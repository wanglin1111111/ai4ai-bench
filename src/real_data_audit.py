"""
AI4AI-Bench 真实数据审计分析 v2（非循环版）

数据源: Harness-R1 官方 heldout_generalization/results.json（真实实验数据）

v1 教训: delta = rescued - regressed, 故 "delta>0 AND ratio<1" 与 ground truth
        数学等价, 100% 准确率是循环论证。v2 修正:
        审计门只能观测到 held-out 的一个随机子集(audit budget, 如 50 任务),
        由子集估计净效果决定接受/拒绝; ground truth 仍为全量 held-out delta。
        抽样不确定性使评估非平凡。另附真实可观测信号(补丁 hook 数)分析。
"""

import json
import numpy as np
from pathlib import Path

# 数据源可用环境变量 HARNESS_R1_RESULTS 覆盖；默认相对本仓库 clone 位置
import os
RESULTS_PATH = Path(os.environ.get("HARNESS_R1_RESULTS",
    "Harness-R1/examples/heldout_generalization/results.json"))
OUT_DIR = Path(__file__).resolve().parent


def load_real_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def estimate_delta_on_sample(tasks: int, rescued: int, regressed: int, n_sample: int, rng) -> tuple:
    """在 held-out 的随机子集上估计净效果。

    全量: rescued 个失败转成功, regressed 个成功转失败。
    子集: 按入样概率 n_sample/tasks 做二项抽样, 子集净效果 d_hat = r - g。
    """
    p_in = n_sample / tasks
    r = rng.binomial(rescued, p_in)
    g = rng.binomial(regressed, p_in)
    return r - g, r, g


def main():
    data = load_real_results()
    rng = np.random.RandomState(42)

    # ---------- 收集 27 个真实补丁 ----------
    patches = []
    for editor_id, editor in data["editors"].items():
        for seed, benchmarks in editor["per_seed"].items():
            for bench, detail in benchmarks.items():
                h = detail["held_out"]
                patches.append({
                    "editor": editor_id, "seed": seed, "benchmark": bench,
                    "valid": detail["valid"],
                    "hooks": len(detail.get("hooks", [])),
                    "delta": h["delta_pass"], "rescued": h["rescued_failures"],
                    "regressed": h["regressed_successes"], "tasks": h["tasks"],
                    "gt_accept": (h["delta_pass"] > 0) if detail["valid"] else False,
                })

    valid = [p for p in patches if p["valid"]]
    invalid = [p for p in patches if not p["valid"]]

    print("=" * 78)
    print("AI4AI-Bench 真实数据审计 v2（非循环：审计门只见子集，ground truth 为全量）")
    print("=" * 78)

    # ---------- 真实信号 1: 补丁 hook 数 vs 净效果 ----------
    print("\n[信号 1] 补丁 hook 数与 held-out 净效果的关系（23 个有效补丁）")
    one_hook = [p["delta"] for p in valid if p["hooks"] == 1]
    multi_hook = [p["delta"] for p in valid if p["hooks"] > 1]
    print(f"  单 hook 补丁 (n={len(one_hook)}): 平均 delta {np.mean(one_hook):+.1f} 任务, "
          f"正增益比例 {np.mean(np.array(one_hook) > 0):.0%}")
    print(f"  多 hook 补丁 (n={len(multi_hook)}): 平均 delta {np.mean(multi_hook):+.1f} 任务, "
          f"正增益比例 {np.mean(np.array(multi_hook) > 0):.0%}")
    # 分编辑器看（避免编辑器能力混淆）
    for ed in ["harness-r1", "qwen3.5-397b", "deepseek-v4-pro"]:
        rows = [p for p in valid if p["editor"] == ed]
        for hks in sorted(set(p["hooks"] for p in rows)):
            sub = [p["delta"] for p in rows if p["hooks"] == hks]
            print(f"    {ed} hooks={hks}: n={len(sub)}, mean delta {np.mean(sub):+.1f}")

    # ---------- 真实信号 2: 破坏规模（regressed）与挽救规模的分离 ----------
    print("\n[信号 2] 挽救/破坏结构（真实退化指纹）")
    for ed in ["harness-r1", "qwen3.5-397b", "deepseek-v4-pro"]:
        rows = [p for p in valid if p["editor"] == ed]
        tot_r = sum(p["rescued"] for p in rows)
        tot_g = sum(p["regressed"] for p in rows)
        worst = min(rows, key=lambda p: p["delta"])
        print(f"  {ed}: 总挽救 {tot_r} / 总破坏 {tot_g} (破坏率 {tot_g/(tot_r+tot_g):.0%}) | "
              f"最差补丁 {worst['benchmark']}/{worst['seed']}: {worst['delta']:+d} "
              f"(r={worst['rescued']}, g={worst['regressed']})")

    # ---------- 非循环审计门: 子集预算扫描 ----------
    print("\n[审计门评估] 审计预算(子集任务数)扫描, 200 次重抽样")
    print(f"{'预算':>6} | {'决策准确率':>8} | {'正确拒绝负补丁':>12} | {'误拒正补丁':>8} | "
          f"{'期望净效果(任务)':>14} | {'相对无审计':>10}")
    print("-" * 78)

    scan_results = {}
    for budget in [20, 50, 100, 200]:
        n_rep = 200
        acc_list, rej_neg_list, wrong_rej_list, net_list = [], [], [], []
        for rep in range(n_rep):
            rep_rng = np.random.RandomState(1000 + rep)
            n_correct, net_effect, rej_neg, wrong_rej = 0, 0, 0, 0
            for p in patches:
                if not p["valid"]:
                    decision = False  # 非法补丁一律拒绝（真实场景验证器先拦截）
                else:
                    d_hat, _, _ = estimate_delta_on_sample(
                        p["tasks"], p["rescued"], p["regressed"], budget, rep_rng)
                    decision = d_hat > 0
                if decision == p["gt_accept"]:
                    n_correct += 1
                if decision:
                    net_effect += p["delta"]
                else:
                    if p["gt_accept"]:
                        wrong_rej += 1
                    else:
                        rej_neg += 1
            acc_list.append(n_correct / len(patches))
            net_list.append(net_effect)
            rej_neg_list.append(rej_neg)
            wrong_rej_list.append(wrong_rej)
        no_audit_net = sum(p["delta"] for p in valid)  # 全接受(含非法补丁按无效=0计)
        scan_results[budget] = {
            "accuracy_mean": round(float(np.mean(acc_list)), 4),
            "accuracy_std": round(float(np.std(acc_list)), 4),
            "expected_net_delta": round(float(np.mean(net_list)), 1),
            "net_delta_std": round(float(np.std(net_list)), 1),
            "avg_correctly_rejected_neg": round(float(np.mean(rej_neg_list)), 2),
            "avg_wrongly_rejected_pos": round(float(np.mean(wrong_rej_list)), 2),
        }
        s = scan_results[budget]
        print(f"{budget:>6} | {s['accuracy_mean']:>7.0%}±{s['accuracy_std']:.0%} | "
              f"{s['avg_correctly_rejected_neg']:>12.2f} | {s['avg_wrongly_rejected_pos']:>8.2f} | "
              f"{s['expected_net_delta']:>+14.1f} | "
              f"{s['expected_net_delta']-no_audit_net:>+10.1f}")

    print(f"\n  (无审计基线净效果 = 全接受 = {no_audit_net:+d} 任务)")

    # ---------- 编辑器级汇总（真实官方数字, 引用用）----------
    editor_summary = {}
    for editor_id, editor in data["editors"].items():
        pooled = editor["pooled_held_out"]
        rows = [p for p in valid if p["editor"] == editor_id]
        editor_summary[editor_id] = {
            "display_name": editor["display_name"],
            "valid_patches": editor["valid_patches"],
            "pooled_delta_pp_mean": pooled["delta_pass_rate_pp_mean"],
            "pooled_delta_pp_std": pooled["delta_pass_rate_pp_sample_std"],
            "per_bench_pp": {b: v["delta_pass_rate_pp_mean"]
                             for b, v in editor["per_benchmark"].items()},
            "n_positive": sum(1 for p in rows if p["delta"] > 0),
            "n_negative": sum(1 for p in rows if p["delta"] < 0),
        }

    report = {
        "source": "Harness-R1 official heldout_generalization results.json (REAL published data)",
        "meta": {
            "protocol": data["protocol"],
            "target_agent": data["target_agent"],
            "held_out_tasks": data["held_out_tasks"],
            "seeds": data["seeds"],
        },
        "editor_summary": editor_summary,
        "signal_hooks": {
            "one_hook_mean_delta": round(float(np.mean(one_hook)), 2),
            "multi_hook_mean_delta": round(float(np.mean(multi_hook)), 2),
        },
        "gate_budget_scan": scan_results,
        "no_audit_baseline_net_delta": no_audit_net,
        "note": ("v1 的 100% 准确率为循环论证(delta=rescued-regressed); "
                 "v2 审计门仅观测随机子集, ground truth 为全量 held-out, 非循环。"),
    }
    out = OUT_DIR / "real_data_audit_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
