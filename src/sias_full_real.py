"""
真实版 SIAS 全量计算（病灶2修复）
对 27 个真实补丁计算完整 SIAS 分数（含实测复杂度项），检验完整式与实际决策的一致性。

复杂度项 C(p) 实测自补丁 JSON 的 code 字段：
  modified_lines = 全 actions 的 code 总行数（新增即全部写入 runtime）
  added_lines    = 非空非注释代码行
  deleted_lines  = 0（add_code_hook 语义：只增不改）
奖励项来自官方 results.json 的 held_out 真实记录。
"""
import json
import numpy as np
from pathlib import Path

import os
BASE = Path(os.environ.get("HARNESS_R1_RESULTS",
    "Harness-R1/examples/heldout_generalization"))
OUT = Path(__file__).resolve().parent / "sias_full_real_results.json"

ALPHA, BETA, GAMMA = 1.0, 0.3, 0.0005


def patch_complexity(patch: dict) -> dict:
    total_lines, code_lines = 0, 0
    for act in patch.get("actions", []):
        code = act.get("code", "")
        lines = code.splitlines()
        total_lines += len(lines)
        code_lines += sum(1 for l in lines
                          if l.strip() and not l.strip().startswith("#"))
    return {"modified_lines": total_lines, "added_lines": code_lines, "deleted_lines": 0}


def sias_full(delta_pp: float, overfit: float, comp: dict) -> float:
    C = (comp["modified_lines"] * GAMMA
         + comp["added_lines"] * GAMMA * 0.6
         + comp["deleted_lines"] * GAMMA * 0.4)
    return ALPHA * delta_pp - BETA * overfit - C


def main():
    rj = json.load(open(BASE / "results.json", encoding="utf-8"))
    rows = []
    for editor_id, editor in rj["editors"].items():
        for seed, benchmarks in editor["per_seed"].items():
            for bench, detail in benchmarks.items():
                h = detail["held_out"]
                if not detail["valid"]:
                    rows.append({
                        "editor": editor_id, "seed": seed, "benchmark": bench,
                        "valid": False, "sias_full": None, "sign_agrees_gt": None,
                        "note": detail.get("validation_error", "")[:60],
                    })
                    continue
                pf = BASE / detail["patch_file"]
                comp = patch_complexity(json.load(open(pf, encoding="utf-8")))
                # delta 以 pp 计（与论文一致）；本数据无 train 项，overfit 项=0（纯 hold-out 审计）
                delta_pp = h["delta_pass"] / h["tasks"] * 100
                s = sias_full(delta_pp, 0.0, comp)
                rows.append({
                    "editor": editor_id, "seed": seed, "benchmark": bench,
                    "valid": True,
                    "modified_lines": comp["modified_lines"],
                    "added_lines": comp["added_lines"],
                    "delta_pp": round(delta_pp, 2),
                    "complexity_penalty": round(
                        comp["modified_lines"] * GAMMA + comp["added_lines"] * GAMMA * 0.6, 4),
                    "sias_full": round(s, 3),
                    "sign_agrees_gt": (s > 0) == (h["delta_pass"] > 0),
                })

    valid = [r for r in rows if r["valid"]]
    agree = sum(1 for r in valid if r["sign_agrees_gt"])

    # 决策翻转检查：复杂度惩罚是否改变过任何决策？
    flips = [r for r in valid
             if (r["sias_full"] > 0) != (r["delta_pp"] > 0)]

    print("=" * 72)
    print("完整 SIAS 全量计算（27 真实补丁，复杂度项实测自 patch JSON code 字段）")
    print("=" * 72)
    print(f"\n有效补丁 {len(valid)} 个；SIAS_full 符号与真实净效果符号一致: {agree}/{len(valid)}")
    print(f"复杂度惩罚导致决策翻转: {len(flips)} 个")
    for r in flips:
        print(f"  翻转: {r['editor']}/{r['seed']}/{r['benchmark']} "
              f"delta={r['delta_pp']}pp lines={r['modified_lines']} SIAS={r['sias_full']}")

    comp_stats = {
        "mean_modified_lines": round(float(np.mean([r["modified_lines"] for r in valid])), 1),
        "max_modified_lines": max(r["modified_lines"] for r in valid),
        "mean_complexity_penalty": round(float(np.mean(
            [r["complexity_penalty"] for r in valid])), 4),
    }
    print(f"\n复杂度分布: 平均 {comp_stats['mean_modified_lines']} 行, "
          f"最大 {comp_stats['max_modified_lines']} 行, "
          f"平均惩罚 {comp_stats['mean_complexity_penalty']}")

    # 关键实证：复杂度项量级 vs 净效果量级
    print(f"\n最小正 delta_pp: {min(r['delta_pp'] for r in valid if r['delta_pp']>0):.2f}pp "
          f"vs 最大复杂度惩罚 {max(r['complexity_penalty'] for r in valid):.4f}"
          f" → 复杂度项在本数据上{(chr(19981))}足以翻转任何决策" if not flips else "")

    summary = {
        "n_valid": len(valid),
        "sign_agreement": agree,
        "n_decision_flips": len(flips),
        "flips": flips,
        "complexity_stats": comp_stats,
        "note": ("纯 hold-out 审计场景无 train 项，overfit 项=0；"
                 "复杂度惩罚最大量级远小于最小正增益，本数据上完整式与符号门决策等价"),
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {OUT}")


if __name__ == "__main__":
    main()
