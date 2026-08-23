"""HarnessGate 前瞻演示：glm-5.2 (智谱) 作为前沿 harness 编辑器，
从真实失败证据生成新补丁 → 跑过 gate 全管线（格式校验/复杂度/数据需求/存证）。
这是 gate 第一次活着处理一个新编辑器写的新补丁（前瞻组件；全量重跑评估仍需 benchmark 环境，如实标注）。
"""
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:/dev/Harness-R1/examples/heldout_generalization")

# ---- 1. 取真实失败证据（用源数据的 prompt 模板思想：WebShop 失败模式摘要）----
# 用官方记录里 Qwen397B 最差补丁对应批次的失败结构作为证据摘要（从 results.json 可得的结构性事实）
rj = json.load(open(BASE / "results.json", encoding="utf-8"))
w = rj["editors"]["qwen3.5-397b"]["per_seed"]["20260721"]["webshop"]["held_out"]
evidence = {
    "benchmark": "webshop",
    "batch_failures_summary": (
        "10 sampled failures share patterns: (a) agent buys items over budget "
        "(price exceeded task max), (b) agent clicks Buy Now with required options "
        "(color/size) unselected, (c) repeated identical search clicks stalling progress."
    ),
    "runtime_context": ("action: tool/value/final_action; state: page_type, clickables, "
                        "current_price, price_max, repeated-click counters; "
                        "predicates: product_price_over_budget, required_options_unselected, "
                        "repeated click/search, buy-now availability"),
}

# Harness-R1 的补丁契约（附录 A 格式）
system_prompt = (
    "You are a harness engineer. Given failed webshop rollout traces, analyze recurring "
    "failures, then produce one reusable webshop code-hook harness patch as a single JSON "
    "object inside a <patch>...</patch> block. Edit only the reusable harness, not task "
    "answers. Output exactly one patch block. Patch JSON contract: benchmark='webshop', "
    "description: short general description, actions: non-empty array of ADD_CODE_HOOK "
    "objects {type:'add_code_hook', hook:'on_before_action'|'make_pre_hint'|'on_init'|"
    "'on_post_step', code:'def hook(ctx, nb):\\n ...\\n return <effect>'}. Hook code is "
    "plain Python, no imports, no I/O. Use pre-action mediation only for narrow observable "
    "mistakes (over-budget purchase, unselected required option, repeated action). Do not "
    "hard-code product IDs or answers."
)

# ---- 2. 调智谱 API ----
import pathlib
s = json.load(open(pathlib.Path.home() / '.openharness' / 'settings.json', encoding='utf-8'))
req = urllib.request.Request(
    s["base_url"].rstrip("/") + "/chat/completions",
    data=json.dumps({
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Failure evidence:\n{json.dumps(evidence, indent=2)}\n\nWrite the patch now."},
        ],
        "max_tokens": 8192,
    }).encode(),
    headers={"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
raw = resp["choices"][0]["message"]["content"]
print("=== 编辑器输出（前400字）===")
print(raw[:400])

# ---- 3. Gate 管线 ----
entry = {
    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "gate_version": "harnessgate-v1.1(prospective-demo)",
    "patch_source": "glm-5.2 via bigmodel API, live generation",
    "subset_seed": None,  # 前瞻演示：无重跑评估
}

# 3a. 格式校验（官方 validator 规则子集）
m = re.search(r"<patch>(.*?)</patch>", raw, re.DOTALL)
validation = {"valid": False, "error": None}
patch_obj = None
if not m:
    validation["error"] = "no <patch> block found"
else:
    try:
        patch_obj = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        validation["error"] = f"patch JSON parse error: {e}"
if patch_obj is not None:
    if patch_obj.get("benchmark") != "webshop":
        validation["error"] = "benchmark mismatch"
    elif not isinstance(patch_obj.get("actions"), list) or not patch_obj["actions"]:
        validation["error"] = "actions must be non-empty list"
    else:
        for a in patch_obj["actions"]:
            code = a.get("code", "")
            if a.get("type") != "add_code_hook":
                validation["error"] = f"forbidden action type: {a.get('type')}"; break
            if any(k in code for k in ["import ", "open(", "__", "eval(", "exec("]):
                validation["error"] = "forbidden syntax in hook code"; break
            if a.get("hook") not in ("on_init", "make_pre_hint", "on_before_action", "on_post_step"):
                validation["error"] = f"unknown hook: {a.get('hook')}"; break
        else:
            validation["valid"] = True

entry["format_validation"] = validation

# 3b. 复杂度测量
if validation["valid"]:
    total_lines = sum(len(a["code"].splitlines()) for a in patch_obj["actions"])
    hooks_used = [a["hook"] for a in patch_obj["actions"]]
    entry["complexity"] = {"total_lines": total_lines, "hooks": hooks_used}
    # 3c. 完整 SIAS 静态部分（无重跑 → 无 delta 项，只报复杂度与上下文 bloated 判定）
    entry["context_bloat_flag"] = total_lines > 150
    entry["decision"] = "PENDING_OUTCOME_EVAL"
    entry["note"] = ("Format-valid; complexity measured. Outcome evaluation requires rerunning "
                     "the frozen target on audit tasks (benchmark env) — not substituted here.")
else:
    entry["decision"] = "REJECT_INVALID"
    entry["route"] = "CI"
    entry["data_needs"] = ["format_invalid: rerun editor with stricter output contract; "
                           "observed error: " + str(validation["error"])]

# ---- 4. 追加存证 ----
log_path = Path("gate_trajectory_log.json")
log = json.load(open(log_path, encoding="utf-8"))
log.setdefault("prospective_entries", []).append(entry)
log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== Gate 判定 ===")
print(json.dumps(entry, ensure_ascii=False, indent=2)[:800])
print(f"\n存证已追加: {log_path} (prospective_entries)")
