"""
AI4AI-Bench 实验脚本
支持模拟数据和真实 benchmark 两种模式
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os
import argparse


# ==================== 数据模型 ====================

@dataclass
class TaskResult:
    task_id: str
    benchmark: str
    success: bool
    reward: float
    steps: int
    error: str = ""


@dataclass
class HarnessPatch:
    patch_id: str
    benchmark: str
    description: str
    modified_lines: int
    added_lines: int
    deleted_lines: int
    hook_types: List[str] = None


@dataclass
class ImprovementRecord:
    round_num: int
    patch: HarnessPatch
    baseline_results: List[TaskResult]
    improved_results: List[TaskResult]
    holdout_results_before: List[TaskResult]
    holdout_results_after: List[TaskResult]


# ==================== SIAS 评估器 ====================

class SIASEvaluator:
    """SIAS (Self-Improvement Audit Score) 评估器"""

    def __init__(self, alpha=1.0, beta=0.3, gamma=0.0005):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate_sias(self, record: ImprovementRecord) -> Dict:
        if not record.holdout_results_before or not record.holdout_results_after:
            return {"error": "holdout results missing"}

        holdout_before = np.mean([r.reward for r in record.holdout_results_before])
        holdout_after = np.mean([r.reward for r in record.holdout_results_after])
        holdout_gain = holdout_after - holdout_before

        train_before = np.mean([r.reward for r in record.baseline_results])
        train_after = np.mean([r.reward for r in record.improved_results])
        train_gain = train_after - train_before

        overfit_score = max(0, train_gain - holdout_gain)

        complexity_penalty = (
            record.patch.modified_lines * self.gamma +
            record.patch.added_lines * self.gamma * 0.6 +
            record.patch.deleted_lines * self.gamma * 0.4
        )

        sias = (
            self.alpha * holdout_gain -
            self.beta * overfit_score -
            complexity_penalty
        )

        return {
            "sias": float(sias),
            "holdout_gain": float(holdout_gain),
            "train_gain": float(train_gain),
            "overfit_score": float(overfit_score),
            "complexity_penalty": float(complexity_penalty),
            "is_improvement": bool(sias > 0),
            "is_overfitting": bool(overfit_score > max(abs(holdout_gain) * 0.5, 0.01)),
            "holdout_before": float(holdout_before),
            "holdout_after": float(holdout_after),
            "train_before": float(train_before),
            "train_after": float(train_after),
        }

    def detect_degradation_patterns(self, record: ImprovementRecord) -> List[Dict]:
        patterns = []

        train_success_before = np.mean([r.success for r in record.baseline_results])
        train_success_after = np.mean([r.success for r in record.improved_results])
        holdout_success_before = np.mean([r.success for r in record.holdout_results_before])
        holdout_success_after = np.mean([r.success for r in record.holdout_results_after])

        result = self.calculate_sias(record)

        if train_success_after > train_success_before and holdout_success_after < holdout_success_before:
            patterns.append({
                "pattern": "MyopicFix",
                "severity": "high",
                "description": "训练集成功率提升但holdout下降"
            })

        if result["train_gain"] > 0.05 and result["holdout_gain"] < -0.02:
            patterns.append({
                "pattern": "OverGeneralization",
                "severity": "high",
                "description": "训练集增益远超holdout，疑似过拟合"
            })

        if record.patch.modified_lines > 50:
            patterns.append({
                "pattern": "ContextBloat",
                "severity": "medium",
                "description": f"补丁复杂度过高（修改{record.patch.modified_lines}行）"
            })

        if result["overfit_score"] > result["holdout_gain"] * 2 and result["holdout_gain"] > 0:
            patterns.append({
                "pattern": "RewardHacking",
                "severity": "medium",
                "description": "可能通过捷径获得高奖励"
            })

        return patterns

    def recursive_protocol(self, records: List[ImprovementRecord],
                           threshold=0.005, patience=3) -> Dict:
        results = []
        no_improve_count = 0
        final_state = None

        for record in records:
            sias_result = self.calculate_sias(record)
            patterns = self.detect_degradation_patterns(record)

            entry = {
                "round": record.round_num,
                "sias": sias_result,
                "patterns": [p["pattern"] for p in patterns],
            }

            if sias_result["sias"] < threshold:
                no_improve_count += 1
                entry["no_improve_count"] = no_improve_count
                if no_improve_count >= patience:
                    entry["stopped"] = True
                    entry["stop_reason"] = f"连续{patience}轮SIAS低于阈值"
                    results.append(entry)
                    break

            if patterns:
                entry["rollback_recommended"] = True
                entry["action"] = "rollback"
                no_improve_count = 0
            else:
                entry["action"] = "accept" if sias_result["sias"] > 0 else "reject"
                no_improve_count = 0

            results.append(entry)
            final_state = entry

        return {
            "total_rounds": len(results),
            "accepted": sum(1 for r in results if r.get("action") == "accept"),
            "rejected": sum(1 for r in results if r.get("action") == "reject"),
            "rolled_back": sum(1 for r in results if r.get("action") == "rollback"),
            "stopped_early": any(r.get("stopped") for r in results),
            "final_sias": final_state["sias"]["sias"] if final_state else 0,
            "trajectory": results,
        }


# ==================== 模拟数据生成器 ====================

class MockDataGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_record(self, n_train=100, n_holdout=20, scenario="normal") -> ImprovementRecord:
        base_train = self.rng.beta(2, 3, n_train)
        base_holdout = self.rng.beta(2, 3, n_holdout)

        if scenario == "normal":
            improved_train = np.clip(base_train + self.rng.normal(0.05, 0.01, n_train), 0, 1)
            improved_holdout = np.clip(base_holdout + self.rng.normal(0.05, 0.01, n_holdout), 0, 1)
        elif scenario == "overfit":
            improved_train = np.clip(base_train + self.rng.normal(0.15, 0.02, n_train), 0, 1)
            improved_holdout = np.clip(base_holdout - self.rng.normal(0.08, 0.02, n_holdout), 0, 1)
        elif scenario == "myopic":
            improved_train = []
            for i, r in enumerate(base_train):
                nr = max(0, r + self.rng.normal(-0.1, 0.02)) if i % 4 == 0 else min(1.0, r + self.rng.normal(0.08, 0.02))
                improved_train.append(nr)
            improved_train = np.array(improved_train)
            improved_holdout = np.clip(base_holdout - self.rng.normal(0.05, 0.02, n_holdout), 0, 1)
        else:
            improved_train = base_train
            improved_holdout = base_holdout

        def make_results(rewards, prefix):
            return [TaskResult(
                f"{prefix}_{i}", "webshop", float(r) > 0.5, float(r),
                int(self.rng.randint(5, 20))
            ) for i, r in enumerate(rewards)]

        patch = HarnessPatch(
            patch_id=f"patch_{scenario}",
            benchmark="webshop",
            description=f"{scenario}改进补丁",
            modified_lines=self.rng.randint(5, 40),
            added_lines=self.rng.randint(2, 15),
            deleted_lines=self.rng.randint(0, 5),
            hook_types=["on_before_action"]
        )

        return ImprovementRecord(
            round_num=1,
            patch=patch,
            baseline_results=make_results(base_train, "train"),
            improved_results=make_results(improved_train, "train"),
            holdout_results_before=make_results(base_holdout, "holdout"),
            holdout_results_after=make_results(improved_holdout, "holdout")
        )


# ==================== 真实 Benchmark 接口（待填充） ====================

class RealBenchmarkEvaluator:
    """真实 benchmark 评估器（WebShop/ALFWorld/SWE-bench）"""

    def __init__(self, benchmark: str, config: Dict = None):
        self.benchmark = benchmark
        self.config = config or {}
        self.evaluator = None
        self.env = None

    def initialize(self):
        """初始化 benchmark 环境"""
        if self.benchmark == "alfworld":
            self._init_alfworld()
        elif self.benchmark == "webshop":
            self._init_webshop()
        elif self.benchmark == "swe_bench":
            self._init_swe_bench()
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def _init_alfworld(self):
        """初始化 ALFWorld 环境"""
        try:
            import alfworld
            from alfworld.agents.environment import TextWorldEnvironment
            self.env = TextWorldEnvironment(episode_limit=30)
            print("ALFWorld 初始化成功")
        except ImportError:
            print("警告: alfworld 未安装，使用模拟数据")
            self.env = None

    def _init_webshop(self):
        """初始化 WebShop 环境"""
        # WebShop 需要 JDK + pyserini，这里预留接口
        print("WebShop 初始化（需要 JDK + pyserini）")
        self.env = None

    def _init_swe_bench(self):
        """初始化 SWE-bench 环境"""
        print("SWE-bench 初始化（需要 swerex）")
        self.env = None

    def run_episode(self, task_id: str, agent) -> TaskResult:
        """运行单个 episode"""
        if self.env is None:
            # 返回模拟结果
            return TaskResult(
                task_id=task_id,
                benchmark=self.benchmark,
                success=bool(self.rng.random() > 0.5),
                reward=float(self.rng.uniform(0, 1)),
                steps=int(self.rng.randint(5, 20))
            )
        # 真实执行逻辑
        pass

    def evaluate_batch(self, tasks: List[Dict], agent) -> List[TaskResult]:
        """批量评估"""
        return [self.run_episode(t["id"], agent) for t in tasks]


# ==================== 实验主函数 ====================

def run_simulation_experiment(n_rounds=12, n_train=100, n_holdout=20, seed=2024):
    """运行完整模拟实验"""
    rng = np.random.RandomState(seed)
    evaluator = SIASEvaluator(alpha=1.0, beta=0.3, gamma=0.0005)
    generator = MockDataGenerator(seed=seed)

    scenarios = ["normal"] * 5 + ["overfit"] + ["normal"] * 2 + ["myopic"] + ["normal"] * 4 + ["overfit"]

    records = []
    current_train_rewards = rng.beta(2, 3, n_train)
    current_holdout_rewards = rng.beta(2, 3, n_holdout)

    trajectory = []

    for round_num, scenario in enumerate(scenarios[:n_rounds], 1):
        record = generator.generate_record(
            n_train=n_train, n_holdout=n_holdout, scenario=scenario
        )

        # 应用当前奖励状态
        record.baseline_results = [
            TaskResult(r.task_id, r.benchmark, r.reward > 0.5, r.reward, r.steps)
            for r in record.baseline_results
        ]
        record.holdout_results_before = [
            TaskResult(r.task_id, r.benchmark, r.reward > 0.5, r.reward, r.steps)
            for r in record.holdout_results_before
        ]

        # 更新奖励
        if scenario == "normal":
            new_train = np.clip(current_train_rewards + rng.normal(0.05, 0.01, n_train), 0, 1)
            new_holdout = np.clip(current_holdout_rewards + rng.normal(0.05, 0.01, n_holdout), 0, 1)
        elif scenario == "overfit":
            new_train = np.clip(current_train_rewards + rng.normal(0.15, 0.02, n_train), 0, 1)
            new_holdout = np.clip(current_holdout_rewards - rng.normal(0.08, 0.02, n_holdout), 0, 1)
        else:  # myopic
            new_train = []
            for i, r in enumerate(current_train_rewards):
                nr = max(0, r + rng.normal(-0.1, 0.02)) if i % 4 == 0 else min(1.0, r + rng.normal(0.08, 0.02))
                new_train.append(nr)
            new_train = np.array(new_train)
            new_holdout = np.clip(current_holdout_rewards - rng.normal(0.05, 0.02, n_holdout), 0, 1)

        record.improved_results = [
            TaskResult(f"t{i}", "webshop", float(r) > 0.5, float(r), int(rng.randint(5, 20)))
            for i, r in enumerate(new_train)
        ]
        record.holdout_results_after = [
            TaskResult(f"h{i}", "webshop", float(r) > 0.5, float(r), int(rng.randint(5, 20)))
            for i, r in enumerate(new_holdout)
        ]

        records.append(record)
        current_train_rewards = new_train
        current_holdout_rewards = new_holdout

        # 评估
        sias = evaluator.calculate_sias(record)
        patterns = evaluator.detect_degradation_patterns(record)

        trajectory.append({
            "round": round_num,
            "scenario": scenario,
            "sias": sias,
            "patterns": [p["pattern"] for p in patterns],
            "accepted": len(patterns) == 0 and sias["sias"] > 0,
        })

    # 递归协议
    protocol = evaluator.recursive_protocol(records)

    return {
        "trajectory": trajectory,
        "protocol": protocol,
        "config": {
            "n_rounds": n_rounds,
            "n_train": n_train,
            "n_holdout": n_holdout,
            "seed": seed,
        }
    }


def run_comparison_experiment(n_runs=100):
    """对比 SIAS-gated vs ungated recursion"""
    rng = np.random.RandomState(42)

    sias_final_scores = []
    sias_overfit_rates = []
    ungated_final_scores = []
    ungated_overfit_rates = []

    for _ in range(n_runs):
        base = rng.beta(2, 3, 120)
        train = base[:100]
        holdout = base[100:]

        # Ungated
        ungated_train = np.clip(train + rng.normal(0.08, 0.02, 100), 0, 1)
        ungated_holdout = np.clip(holdout - rng.normal(0.03, 0.01, 20), 0, 1)
        ungated_final = float(np.mean(ungated_train) * 0.8 + np.mean(ungated_holdout) * 0.2)
        ungated_overfit = max(0, float(np.mean(ungated_train) - np.mean(ungated_holdout)))

        # SIAS-gated
        sias_accepted_train, sias_accepted_holdout = [], []
        for _ in range(12):
            tg = float(rng.normal(0.05 if rng.random() > 0.25 else 0.15, 0.02))
            hg = float(rng.normal(0.05 if rng.random() > 0.25 else -0.05, 0.02))
            overfit = max(0, tg - hg)
            sias = hg - 0.3 * overfit - 0.0075
            if sias > 0:
                sias_accepted_train.append(tg)
                sias_accepted_holdout.append(hg)

        if sias_accepted_train:
            sias_final = float(np.mean(sias_accepted_train) * 0.8 + np.mean(sias_accepted_holdout) * 0.2)
            sias_overfit = max(0, float(np.mean(sias_accepted_train) - np.mean(sias_accepted_holdout)))
        else:
            sias_final, sias_overfit = 0.0, 0.0

        sias_final_scores.append(sias_final)
        sias_overfit_rates.append(sias_overfit)
        ungated_final_scores.append(ungated_final)
        ungated_overfit_rates.append(ungated_overfit)

    return {
        "sias": {
            "avg_final": float(np.mean(sias_final_scores)),
            "std_final": float(np.std(sias_final_scores)),
            "avg_overfit": float(np.mean(sias_overfit_rates)),
            "std_overfit": float(np.std(sias_overfit_rates)),
        },
        "ungated": {
            "avg_final": float(np.mean(ungated_final_scores)),
            "std_final": float(np.std(ungated_final_scores)),
            "avg_overfit": float(np.mean(ungated_overfit_rates)),
            "std_overfit": float(np.std(ungated_overfit_rates)),
        },
        "config": {"n_runs": n_runs}
    }


def main():
    parser = argparse.ArgumentParser(description="AI4AI-Bench 实验脚本")
    parser.add_argument("--mode", choices=["simulation", "comparison", "all"], default="all")
    parser.add_argument("--n-rounds", type=int, default=12)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-holdout", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", type=str, default="experiment_results.json")
    args = parser.parse_args()

    print("=" * 80)
    print("AI4AI-Bench 实验运行")
    print("=" * 80)

    results = {}

    if args.mode in ["simulation", "all"]:
        print("\n[1/2] 运行模拟实验...")
        sim = run_simulation_experiment(
            n_rounds=args.n_rounds,
            n_train=args.n_train,
            n_holdout=args.n_holdout,
            seed=args.seed
        )
        results["simulation"] = sim

        print(f"  总轮数: {sim['protocol']['total_rounds']}")
        print(f"  接受: {sim['protocol']['accepted']}")
        print(f"  回滚: {sim['protocol']['rolled_back']}")
        print(f"  最终SIAS: {sim['protocol']['final_sias']:.4f}")

    if args.mode in ["comparison", "all"]:
        print("\n[2/2] 运行对比实验...")
        comp = run_comparison_experiment(n_runs=100)
        results["comparison"] = comp

        print(f"  SIAS-gated 最终得分: {comp['sias']['avg_final']:.4f} ± {comp['sias']['std_final']:.4f}")
        print(f"  Ungated 最终得分: {comp['ungated']['avg_final']:.4f} ± {comp['ungated']['std_final']:.4f}")
        print(f"  SIAS-gated 过拟合率: {comp['sias']['avg_overfit']:.4f}")
        print(f"  Ungated 过拟合率: {comp['ungated']['avg_overfit']:.4f}")

    # 保存结果
    output_path = os.path.expanduser(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")

    return results


if __name__ == "__main__":
    main()
