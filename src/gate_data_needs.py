"""
Data-Need Estimator（反向数据需求层）+ 决策存证 trajectory log
圆桌吸收项 A + C 的原型实现。

核心思想（圆桌 30:42）："大模型解决的第一个问题是告诉我需要什么数据"。
gate 拒绝一个补丁时不该只说 no——应输出：
  1. 这个决策的置信度有多低（抽样不确定性）
  2. 要把决策置信度提到目标水平，还需要多少审计任务（n 需求估计）
  3. 需要什么类型的数据（哪些失败模式证据不足）

同时每个决策写入 append-only trajectory log（决策存证）：
  gate 版本 / 审计子集种子 / 预算 / 观测值 / 决策 / 责任路由 / 时间戳
"""
import json
import math
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:/dev/Harness-R1/examples/heldout_generalization")
OUT_DIR = Path(r"C:/Users/22812/OneDrive/Desktop/本机记忆")

GATE_VERSION = "sias-gate-v1.0(subset-sign)"


def one_sided_ci_width(n_tasks: int, delta_hat: float, p_in: float) -> float:
    """审计子集上净效果估计的近似 95% 单侧置信半宽。

    观测 r ~ Bin(rescued, p_in), g ~ Bin(regressed, p_in)
    d_hat = r - g, Var(d_hat) ≈ r + g (泊松近似，两者独立)
    """
    # 用 d_hat 的两个分支估计 r+g 的量级：保守取 |d_hat| + 2*sqrt(|d_hat|) 上界
    rg_mag = abs(delta_hat) + 2 * math.sqrt(max(abs(delta_hat), 1))
    return 1.96 * math.sqrt(max(rg_mag, 1))


def estimate_n_need(d_hat: float, n_used: int, target_conf: float = 0.9) -> dict:
    """估计把决策置信度提到目标水平所需的审计任务数。

    简化模型：d_hat 的标准误 se ∝ sqrt(n_used)，
    置信 z 分数 = |d_hat| / se；要 z >= z_target 需 n_need = n_used * (z_target * se / |d_hat|)^2
    """
    if d_hat == 0:
        return {"n_need": None, "reason": "zero observed effect; need qualitative failure analysis, not more sampling"}
    se = math.sqrt(max(abs(d_hat), 1))
    z = abs(d_hat) / se
    z_target = 1.645 if target_conf == 0.9 else 2.326
    if z >= z_target:
        return {"n_need": 0, "reason": f"sufficient (z={z:.2f} >= {z_target})"}
    scale = (z_target / z) ** 2
    return {"n_need": int(min(n_used * scale, 490)), "z_current": round(z, 2),
            "reason": f"z={z:.2f} < {z_target}; scaling budget by {scale:.1f}x"}


def classify_data_need(patch: dict, d_hat: int, r: int, g: int) -> list:
    """根据观测结构输出缺什么类型的数据（定性需求清单）。"""
    needs = []
    if r == 0 and g == 0:
        needs.append("no_signal_in_subset: subset contains no flipped tasks — "
                     "sample tasks from the benchmark's failure-dense regions (where baseline failed)")
    if d_hat <= 0 and r > 0:
        needs.append(f"mixed_effect: rescued({r}) ~ regressed({g}) — need per-task trajectory diffs "
                     "to identify which baseline successes regressed and why")
    if patch["benchmark"] == "webshop" and d_hat < 0:
        needs.append("webshop_negative: prior real failures concentrated in over-aggressive action "
                     "blocking — need failure packets of blocked-but-correct actions")
    if patch["benchmark"] == "alfworld" and d_hat < 0:
        needs.append("alfworld_negative: prior regressions concentrated in multi-hook patches — "
                     "need hook-trigger logs to attribute damage to a specific hook")
    if not needs:
        needs.append(f"low_power: effect direction ({d_hat:+d}) uncertain at this budget — "
                     "increase audit sample before deciding")
    return needs


def run_gate_with_data_needs(budget: int = 50, seed: int = 2024) -> dict:
    rj = json.load(open(BASE / "results.json", encoding="utf-8"))
    rng = np.random.RandomState(seed)

    log_entries = []
    n_rejected_lowconf = 0
    data_need_issued = 0

    for ed, editor in rj["editors"].items():
        for s, bms in editor["per_seed"].items():
            for bench, d in bms.items():
                h = d["held_out"]
                entry = {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "gate_version": GATE_VERSION,
                    "patch": f"{ed}/{s}/{bench}",
                    "budget": budget, "subset_seed": seed,
                }
                if not d["valid"]:
                    entry.update({"decision": "REJECT_INVALID", "route": "CI",
                                  "d_hat": None, "data_needs": ["format_invalid: rerun editor with stricter output contract"]})
                else:
                    p_in = budget / h["tasks"]
                    r = int(rng.binomial(h["rescued_failures"], p_in))
                    g = int(rng.binomial(h["regressed_successes"], p_in))
                    d_hat = r - g
                    entry["observed"] = {"r": r, "g": g, "d_hat": d_hat}
                    if d_hat > 0:
                        entry.update({"decision": "ACCEPT", "route": "none"})
                    else:
                        n = estimate_n_need(d_hat, budget)
                        needs = classify_data_need(
                            {"benchmark": bench}, d_hat, r, g)
                        entry.update({"decision": "REJECT", "route": "harness-engineer",
                                      "confidence": n.get("z_current"),
                                      "n_more_tasks_needed": n.get("n_need"),
                                      "data_needs": needs})
                        n_rejected_lowconf += 1 if (n.get("n_need") or 0) > 0 else 0
                        data_need_issued += len(needs)
                log_entries.append(entry)

    return {
        "gate_version": GATE_VERSION,
        "budget": budget, "subset_seed": seed,
        "n_decisions": len(log_entries),
        "n_accept": sum(1 for e in log_entries if e["decision"] == "ACCEPT"),
        "n_reject": sum(1 for e in log_entries if e["decision"] == "REJECT"),
        "n_reject_low_conf_with_n_estimate": n_rejected_lowconf,
        "n_data_need_items_issued": data_need_issued,
        "trajectory_log": log_entries,
        "audit_note": ("append-only: every entry carries gate_version+subset_seed+budget; "
                       "rollback/override requires appending a new entry, never editing"),
    }


def main():
    result = run_gate_with_data_needs(budget=50)

    print("=" * 74)
    print("Data-Need Estimator + 决策存证 trajectory log（gate v1.0，预算 50）")
    print("=" * 74)
    print(f"决策: {result['n_accept']} ACCEPT / {result['n_reject']} REJECT "
          f"(低置信拒绝且给出 n 需求估计: {result['n_reject_low_conf_with_n_estimate']})")
    print(f"发出的数据需求项: {result['n_data_need_items_issued']} 条\n")

    print("--- REJECT 决策示例（含数据需求）---")
    shown = 0
    for e in result["trajectory_log"]:
        if e["decision"] == "REJECT" and e.get("data_needs") and shown < 3:
            obs = e.get("observed", {})
            print(f"\n{e['patch']} | gate={e['gate_version']} seed={e['subset_seed']}")
            print(f"  观测: r={obs.get('r')}, g={obs.get('g')}, d_hat={obs.get('d_hat')}")
            if e.get("n_more_tasks_needed") is not None:
                print(f"  还需审计任务数: {e['n_more_tasks_needed']} (z={e.get('confidence')})")
            for need in e["data_needs"]:
                print(f"  [数据需求] {need}")
            shown += 1

    # 存证日志（append-only 示范）
    log_path = OUT_DIR / "gate_trajectory_log.json"
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\ntrajectory log 已存: {log_path}")
    print("审计不变量: 回滚/覆盖=追加新条目，永不改旧条目")


if __name__ == "__main__":
    main()
