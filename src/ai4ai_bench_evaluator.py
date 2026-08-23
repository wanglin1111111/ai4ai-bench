from pathlib import Path
"""
AI4AI-Bench Mock Evaluator
用于在没有真实 benchmark 环境时验证 SIAS 指标
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import os

@dataclass
class TaskResult:
    task_id: str
    benchmark: str
    success: bool
    reward: float
    steps: int
    error: str = ""
    trajectory_length: int = 0

@dataclass
class HarnessPatch:
    patch_id: str
    benchmark: str
    description: str
    actions: List[Dict]
    modified_lines: int
    added_lines: int
    deleted_lines: int
    timestamp: str = ""

@dataclass
class ImprovementRecord:
    round_num: int
    patch: HarnessPatch
    baseline_results: List[TaskResult]
    improved_results: List[TaskResult]
    holdout_results_before: List[TaskResult]
    holdout_results_after: List[TaskResult]
    metadata: Dict = None

class SIASEvaluator:
    """SIAS (Self-Improvement Audit Score) 评估器"""
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def calculate_sias(self, record: ImprovementRecord) -> Dict:
        """计算 SIAS 分数"""
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
            record.patch.modified_lines * 0.1 +
            record.patch.added_lines * 0.05 +
            record.patch.deleted_lines * 0.02
        )
        
        sias = (
            self.alpha * holdout_gain -
            self.beta * overfit_score -
            self.gamma * complexity_penalty
        )
        
        return {
            "sias": float(sias),
            "holdout_gain": float(holdout_gain),
            "train_gain": float(train_gain),
            "overfit_score": float(overfit_score),
            "complexity_penalty": float(complexity_penalty),
            "is_improvement": bool(sias > 0),
            "is_overfitting": bool(overfit_score > max(holdout_gain * 0.5, 0.01)),
            "holdout_before": float(holdout_before),
            "holdout_after": float(holdout_after),
            "train_before": float(train_before),
            "train_after": float(train_after),
        }
    
    def detect_degradation_patterns(self, record: ImprovementRecord) -> List[str]:
        """检测退化模式"""
        patterns = []
        
        # Myopic Fix: 训练集提升但 holdout 下降
        train_success_before = np.mean([r.success for r in record.baseline_results])
        train_success_after = np.mean([r.success for r in record.improved_results])
        holdout_success_before = np.mean([r.success for r in record.holdout_results_before])
        holdout_success_after = np.mean([r.success for r in record.holdout_results_after])
        
        if train_success_after > train_success_before and holdout_success_after < holdout_success_before:
            patterns.append("MyopicFix")
            
        # Over-Generalization
        result = self.calculate_sias(record)
        if result.get("train_gain", 0) > 0.05 and result.get("holdout_gain", 0) < -0.02:
            patterns.append("OverGeneralization")
            
        # Context Bloat
        if record.patch.modified_lines > 50:
            patterns.append("ContextBloat")
            
        return patterns
    
    def recursive_protocol(self, records: List[ImprovementRecord], threshold=0.01, patience=3) -> Dict:
        """递归改进协议：SIAS-gated 停止准则"""
        results = []
        no_improve_count = 0
        final_state = None
        
        for record in records:
            sias_result = self.calculate_sias(record)
            patterns = self.detect_degradation_patterns(record)
            
            results.append({
                "round": record.round_num,
                "sias": sias_result,
                "patterns": patterns,
                "patch_desc": record.patch.description,
            })
            
            # 停止准则
            if sias_result.get("sias", 0) < threshold:
                no_improve_count += 1
                if no_improve_count >= patience:
                    results[-1]["stopped"] = True
                    results[-1]["stop_reason"] = "SIAS below threshold"
                    break
            
            if patterns:
                results[-1]["rollback_recommended"] = True
                results[-1]["action"] = "rollback"
                no_improve_count = 0
            else:
                no_improve_count = 0
                results[-1]["action"] = "accept"
                
            final_state = results[-1]
            
        return {
            "total_rounds": len(results),
            "accepted": sum(1 for r in results if r.get("action") == "accept"),
            "rolled_back": sum(1 for r in results if r.get("action") == "rollback"),
            "stopped_early": any(r.get("stopped") for r in results),
            "final_sias": final_state["sias"]["sias"] if final_state else 0,
            "trajectory": results,
        }


class MockDataGenerator:
    """生成模拟数据用于测试"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
    def generate_improvement_record(
        self,
        n_train=100,
        n_holdout=20,
        scenario="normal",
        patch_complexity=10
    ) -> ImprovementRecord:
        """生成单轮改进记录"""
        base_success = self.rng.beta(2, 3, n_train + n_holdout)
        
        if scenario == "normal":
            train_improved = base_success[:n_train] + self.rng.normal(0.03, 0.02, n_train)
            holdout_improved = base_success[n_train:] + self.rng.normal(0.03, 0.02, n_holdout)
        elif scenario == "overfit":
            train_improved = base_success[:n_train] + self.rng.normal(0.1, 0.02, n_train)
            holdout_improved = base_success[n_train:] - self.rng.normal(0.03, 0.01, n_holdout)
        elif scenario == "myopic":
            train_success = base_success[:n_train] + self.rng.normal(0.08, 0.02, n_train)
            holdout_success = base_success[n_train:] - self.rng.normal(0.05, 0.01, n_holdout)
            train_improved = np.clip(train_success, 0, 1)
            holdout_improved = np.clip(holdout_success, 0, 1)
        else:
            train_improved = base_success[:n_train]
            holdout_improved = base_success[n_train:]
            
        train_improved = np.clip(train_improved, 0, 1)
        holdout_improved = np.clip(holdout_improved, 0, 1)
        
        baseline = [TaskResult(
            f"task_{i}", "webshop", r > 0.5, r, self.rng.randint(5, 20)
        ) for i, r in enumerate(base_success[:n_train])]
        
        improved = [TaskResult(
            f"task_{i}", "webshop", r > 0.5, r, self.rng.randint(5, 20)
        ) for i, r in enumerate(train_improved)]
        
        holdout_before = [TaskResult(
            f"holdout_{i}", "webshop", r > 0.5, r, self.rng.randint(5, 20)
        ) for i, r in enumerate(base_success[n_train:])]
        
        holdout_after = [TaskResult(
            f"holdout_{i}", "webshop", r > 0.5, r, self.rng.randint(5, 20)
        ) for i, r in enumerate(holdout_improved)]
        
        patch = HarnessPatch(
            patch_id=f"patch_{scenario}",
            benchmark="webshop",
            description=f"{scenario}改进补丁",
            actions=[{"type": "add_code_hook", "hook": "on_before_action"}],
            modified_lines=patch_complexity,
            added_lines=patch_complexity // 2,
            deleted_lines=patch_complexity // 4,
            timestamp=datetime.now().isoformat()
        )
        
        return ImprovementRecord(
            round_num=1,
            patch=patch,
            baseline_results=baseline,
            improved_results=improved,
            holdout_results_before=holdout_before,
            holdout_results_after=holdout_after
        )


def main():
    evaluator = SIASEvaluator()
    generator = MockDataGenerator()
    
    print("=" * 70)
    print("AI4AI-Bench SIAS 评估器测试")
    print("=" * 70)
    
    # 测试1: 正常改进
    print("\n[测试1] 正常改进场景")
    record1 = generator.generate_improvement_record(scenario="normal")
    result1 = evaluator.calculate_sias(record1)
    patterns1 = evaluator.detect_degradation_patterns(record1)
    print(f"SIAS: {result1['sias']:.4f}")
    print(f"Holdout增益: {result1['holdout_gain']:.4f}")
    print(f"训练集增益: {result1['train_gain']:.4f}")
    print(f"退化模式: {patterns1}")
    
    # 测试2: 过拟合
    print("\n[测试2] 过拟合场景")
    record2 = generator.generate_improvement_record(scenario="overfit")
    result2 = evaluator.calculate_sias(record2)
    patterns2 = evaluator.detect_degradation_patterns(record2)
    print(f"SIAS: {result2['sias']:.4f}")
    print(f"过拟合检测: {result2['is_overfitting']}")
    print(f"退化模式: {patterns2}")
    
    # 测试3: Myopic Fix
    print("\n[测试3] Myopic Fix场景")
    record3 = generator.generate_improvement_record(scenario="myopic")
    result3 = evaluator.calculate_sias(record3)
    patterns3 = evaluator.detect_degradation_patterns(record3)
    print(f"SIAS: {result3['sias']:.4f}")
    print(f"退化模式: {patterns3}")
    
    # 测试4: 递归协议
    print("\n[测试4] 递归改进协议")
    records = [
        generator.generate_improvement_record(scenario="normal", patch_complexity=10),
        generator.generate_improvement_record(scenario="overfit", patch_complexity=20),
        generator.generate_improvement_record(scenario="normal", patch_complexity=15),
        generator.generate_improvement_record(scenario="no_change", patch_complexity=5),
    ]
    protocol_result = evaluator.recursive_protocol(records)
    print(f"总轮数: {protocol_result['total_rounds']}")
    print(f"接受: {protocol_result['accepted']}")
    print(f"回滚: {protocol_result['rolled_back']}")
    print(f"最终SIAS: {protocol_result['final_sias']:.4f}")
    
    # 保存测试报告
    report = {
        "test_time": datetime.now().isoformat(),
        "tests": [
            {"name": "normal", "sias": result1["sias"], "patterns": patterns1},
            {"name": "overfit", "sias": result2["sias"], "patterns": patterns2},
            {"name": "myopic", "sias": result3["sias"], "patterns": patterns3},
        ],
        "recursive_protocol": protocol_result
    }
    
    output_path = Path(__file__).parent / "sias_test_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试报告已保存: {output_path}")


if __name__ == "__main__":
    main()
