"""HarnessGate 前瞻演示：glm-5.2 (智谱) 作为前沿 harness 编辑器，
从真实失败证据生成新补丁 → 跑过 gate 全管线（格式校验/复杂度/数据需求/存证）。
这是 gate 第一次活着处理一个新编辑器写的新补丁（前瞻组件；全量重跑评估仍需 benchmark 环境，如实标注）。

用法：
  # 离线自测（无网络、无上游数据、无 API key，只验证 gate 管线）
  python src/prospective_demo.py --offline

  # 用已有的编辑器输出文件跑 gate（不联网）
  python src/prospective_demo.py --raw-file raw_output.txt

  # 在线调用智谱 API（需要 ~/.openharness/settings.json 与上游 results.json）
  export HARNESS_R1_RESULTS=/path/to/Harness-R1/examples/heldout_generalization/results.json
  python src/prospective_demo.py

  # 显式指定路径
  python src/prospective_demo.py --results /path/to/results.json --settings /path/to/settings.json --log /path/to/log.json
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "results" / "gate_trajectory_log.json"
DEFAULT_SETTINGS = Path.home() / ".openharness" / "settings.json"

SYSTEM_PROMPT = (
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

OFFLINE_RAW = (
    "<patch>"
    '{"benchmark": "webshop", "description": "Pre-action guard against over-budget purchase", '
    '"actions": [{"type": "add_code_hook", "hook": "on_before_action", '
    '"code": "def hook(ctx, nb):\\n'
    '    price = ctx.get(\'current_price\', 0)\\n'
    '    limit = ctx.get(\'price_max\', 0)\\n'
    '    if limit and price > limit:\\n'
    '        return {\'effect\': \'block\', \'reason\': \'over budget\'}\\n'
    '    return {\'effect\': \'pass\'}"}]}'
    "</patch>"
)


def load_results_json(path_arg):
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg))
    import os
    if os.environ.get("HARNESS_R1_RESULTS"):
        candidates.append(Path(os.environ["HARNESS_R1_RESULTS"]))
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        "ERROR: 未找到上游 results.json。请用 --results 指定，或设置环境变量 "
        "HARNESS_R1_RESULTS 指向 Harness-R1/examples/heldout_generalization/results.json；"
        "无上游数据时可用 --offline 做管线自测。"
    )


def build_evidence(results_path):
    rj = json.load(open(results_path, encoding="utf-8"))
    w = rj["editors"]["qwen3.5-397b"]["per_seed"]["20260721"]["webshop"]["held_out"]
    return {
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
        "_held_out_preview": str(w)[:200],
    }


def call_editor(settings_path, evidence):
    try:
        s = json.load(open(settings_path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"ERROR: 无法读取 API 配置 {settings_path}: {e}。可用 --raw-file 或 --offline 绕过联网。")
    for key in ("base_url", "api_key"):
        if key not in s:
            raise SystemExit(f"ERROR: 配置缺少字段 {key}: {settings_path}")
    req = urllib.request.Request(
        s["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": "glm-5.2",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Failure evidence:\n{json.dumps(evidence, indent=2)}\n\nWrite the patch now."},
            ],
            "max_tokens": 8192,
        }).encode(),
        headers={"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        return resp["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        raise SystemExit(f"ERROR: 调用智谱 API 失败（网络/超时/鉴权）: {e}。可改用 --raw-file 或 --offline。")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise SystemExit(f"ERROR: API 响应格式异常: {e}")


def run_gate(raw, patch_source):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate_version": "harnessgate-v1.1(prospective-demo)",
        "patch_source": patch_source,
        "subset_seed": None,
    }
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

    if validation["valid"]:
        total_lines = sum(len(a["code"].splitlines()) for a in patch_obj["actions"])
        entry["complexity"] = {"total_lines": total_lines, "hooks": [a["hook"] for a in patch_obj["actions"]]}
        entry["context_bloat_flag"] = total_lines > 150
        entry["decision"] = "PENDING_OUTCOME_EVAL"
        entry["note"] = ("Format-valid; complexity measured. Outcome evaluation requires rerunning "
                         "the frozen target on audit tasks (benchmark env) — not substituted here.")
    else:
        entry["decision"] = "REJECT_INVALID"
        entry["route"] = "CI"
        entry["data_needs"] = ["format_invalid: rerun editor with stricter output contract; "
                               "observed error: " + str(validation["error"])]
    return entry


def append_log(log_path, entry):
    if log_path.is_file():
        log = json.load(open(log_path, encoding="utf-8"))
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = {}
    log.setdefault("prospective_entries", []).append(entry)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="HarnessGate 前瞻演示（支持离线管线自测）")
    ap.add_argument("--offline", action="store_true", help="不联网、不读上游数据，用内置合法样例跑 gate 管线")
    ap.add_argument("--raw-file", help="从文件读取编辑器输出（不联网）")
    ap.add_argument("--results", help="上游 Harness-R1 results.json 路径（默认读 HARNESS_R1_RESULTS）")
    ap.add_argument("--settings", default=str(DEFAULT_SETTINGS), help="API 配置 settings.json 路径")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="存证日志路径（默认 results/gate_trajectory_log.json）")
    args = ap.parse_args()

    if args.offline:
        raw = OFFLINE_RAW
        source = "offline-builtin-sample"
    elif args.raw_file:
        raw = Path(args.raw_file).read_text(encoding="utf-8")
        source = f"raw-file:{args.raw_file}"
    else:
        results_path = load_results_json(args.results)
        evidence = build_evidence(results_path)
        raw = call_editor(Path(args.settings), evidence)
        source = "glm-5.2 via bigmodel API, live generation"

    print("=== 编辑器输出（前400字）===")
    print(raw[:400])
    entry = run_gate(raw, source)
    log_path = Path(args.log)
    append_log(log_path, entry)
    print("\n=== Gate 判定 ===")
    print(json.dumps(entry, ensure_ascii=False, indent=2)[:800])
    print(f"\n存证已追加: {log_path} (prospective_entries)")
    return 0 if entry["format_validation"]["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
