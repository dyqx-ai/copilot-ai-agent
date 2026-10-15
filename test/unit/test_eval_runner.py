"""测试 eval_runner.py — EvalRunner 编排逻辑。

验证 evaluate_full 的三层编排、条件分支（正确停止/HITL）、
compute_summary 的跨用例聚合、以及 Judge 配置校验。
"""

import pytest

from eval.eval_trace import AgentTrace, ToolCallRecord
from eval.eval_runner import EvalRunner, _model_family, validate_judge_config


def _trace(tool_calls=None, stop_reason=None, **kw):
    return AgentTrace(run_id="r", session_id="s", test_case_id="t",
                      task_input="查询", task_output="完成", stop_reason=stop_reason,
                      tool_calls=tool_calls or [], **kw)


def _tool_call(tool_name, params=None, confirmed=False, error=None):
    return ToolCallRecord(step_name="s", tool_name=tool_name, params=params or {},
                          confirmed_by_user=confirmed, error=error)


class TestEvaluateFull:
    """evaluate_full 完整编排"""

    def test_basic_structure(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})])
        result = runner.evaluate_full(trace)
        assert "L1_execution" in result
        assert "L2_safety" in result
        assert "L3_quality" in result
        assert result["run_id"] == "r"
        assert result["task_input"] == "查询"

    def test_l3_skipped_without_judge(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})])
        result = runner.evaluate_full(trace, with_judge=False)
        assert result["L3_quality"]["skipped"] is True
        assert result["L3_quality"]["goal_achievement"]["goal_achievement_score"] is None

    def test_correct_stop_included_when_expected(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})], stop_reason="param_incomplete")
        result = runner.evaluate_full(trace, expected_stop_reason="param_incomplete")
        assert "correct_stop" in result
        assert result["correct_stop"]["correct_stop_rate"] == 1.0

    def test_correct_stop_omitted_without_expected(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})], stop_reason="param_incomplete")
        result = runner.evaluate_full(trace)  # 不传 expected_stop_reason
        assert "correct_stop" not in result

    def test_hitl_intent_included_when_expected(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})])
        trace.feedback_log = [{"intent": "confirm", "patch": {}}]
        result = runner.evaluate_full(trace, expected_intents=["confirm"], expected_patches=[None])
        assert "HITL_intent" in result
        assert result["HITL_intent"]["intent_accuracy"]["intent_accuracy"] == 1.0

    def test_hitl_intent_omitted_without_expected(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})])
        trace.feedback_log = [{"intent": "confirm"}]
        result = runner.evaluate_full(trace)  # 不传 expected_intents
        assert "HITL_intent" not in result

    def test_l1_with_expected_tools(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})])
        result = runner.evaluate_full(trace, expected_tools=["tool1", "tool2"])
        l1 = result["L1_execution"]
        assert "tool_selection" in l1
        assert l1["tool_selection"]["f1"] == pytest.approx(0.6667, abs=0.001)  # TP=1,FP=0,FN=1

    def test_l1_omits_selection_without_expected(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})])
        result = runner.evaluate_full(trace)  # 不传 expected_tools
        assert "tool_selection" not in result["L1_execution"]


class TestEvaluateL1:
    """L1 层条件计算"""

    def test_hallucination_needs_available_tools(self):
        runner = EvalRunner(available_tools={"tool1"})
        trace = _trace([_tool_call("tool1", {"id": 1})])
        l1 = runner.evaluate_l1(trace)
        assert "tool_hallucination" in l1
        assert l1["tool_hallucination"]["hallucination_rate"] == 0.0

    def test_hallucination_omitted_without_available(self):
        runner = EvalRunner()  # available_tools 为空
        trace = _trace([_tool_call("tool1", {"id": 1})])
        l1 = runner.evaluate_l1(trace)
        assert "tool_hallucination" not in l1

    def test_token_cost_needs_llm_calls(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})])
        trace.llm_calls = []
        l1 = runner.evaluate_l1(trace)
        assert "token_cost" not in l1  # llm_calls 为空不计算

    def test_token_cost_with_llm_calls(self):
        from eval.eval_trace import LLMCallRecord
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1", {"id": 1})])
        trace.llm_calls = [LLMCallRecord("s", "qwen-max", "p", "r", 10, 5, 100)]
        l1 = runner.evaluate_l1(trace)
        assert "token_cost" in l1
        assert l1["token_cost"]["total_tokens"] == 15


class TestEvaluateL2:
    """L2 层高风险工具解析"""

    def test_high_risk_resolved_via_name_mapping(self):
        # HIGH_RISK_TOOLS_BY_HUMAN_NAME 含"创建订单"，映射为 tool17
        runner = EvalRunner()
        trace = _trace([_tool_call("tool17", confirmed=False)])
        # 提供 name_mapping 将中文名映射为 name_for_model
        l2 = runner.evaluate_l2(trace, name_mapping={"创建订单": "tool17"})
        hrc = l2["high_risk_confirmation"]
        assert hrc["high_risk_calls_total"] == 1
        assert hrc["missing_confirmation_rate"] == 1.0

    def test_high_risk_no_mapping_no_calls(self):
        # 无 name_mapping 时，高风险工具集为空 → 无高风险调用
        runner = EvalRunner()
        trace = _trace([_tool_call("tool17", confirmed=False)])
        l2 = runner.evaluate_l2(trace)
        assert l2["high_risk_confirmation"]["high_risk_calls_total"] == 0

    def test_permission_violation_with_allowed(self):
        runner = EvalRunner()
        trace = _trace([_tool_call("tool1"), _tool_call("tool2")])
        l2 = runner.evaluate_l2(trace, allowed_tools={"tool1"})
        assert l2["permission_violation"]["permission_violation_rate"] == 0.5


class TestComputeSummary:
    """compute_summary 跨用例聚合"""

    def _make_result(self, f1=None, hallucination=0.0, unconfirmed=0.0,
                     goal=None, expected_stop=False, intent=None):
        result = {
            "L1_execution": {},
            "L2_safety": {},
            "L3_quality": {"skipped": goal is None},
        }
        if f1 is not None:
            result["L1_execution"]["tool_selection"] = {"f1": f1}
        if hallucination is not None:
            result["L1_execution"]["tool_hallucination"] = {"hallucination_rate": hallucination}
        result["L1_execution"]["call_failure"] = {"failure_rate": 0.0}
        result["L1_execution"]["call_redundancy"] = {"redundancy_rate": 0.0}
        result["L1_execution"]["latency"] = {"e2e_latency_ms": 1000}
        result["L2_safety"]["high_risk_confirmation"] = {"missing_confirmation_rate": unconfirmed}
        result["L2_safety"]["permission_violation"] = {"permission_violation_rate": 0.0}
        result["L2_safety"]["audit_trail"] = {"audit_completeness": 1.0}
        if expected_stop:
            result["_case_expected_stop"] = "tool_not_found"
        if goal is not None:
            result["L3_quality"] = {
                "skipped": False,
                "goal_achievement": {"goal_achievement_score": goal},
                "task_success": {"task_success": goal >= 4},
                "factual_accuracy": {"factual_accuracy_score": 5.0},
                "plan_soundness": {"plan_soundness_score": 5.0},
                "evidence_to_action": {"evidence_to_action_score": 5.0},
                "error_diagnosis": {"error_diagnosis_score": 5.0},
                "param_semantic": {"mean_score": 5.0},
                "statement_hallucination": {"hallucination_rate": 0.0},
                "overconfidence": {"overconfidence_score": 5.0},
            }
        if intent is not None:
            result["HITL_intent"] = {
                "intent_accuracy": {"intent_accuracy": intent, "applicable": True},
                "slot_completeness": {"slot_completeness": intent, "applicable": True},
            }
        return result

    def test_empty_results(self):
        summary = EvalRunner().compute_summary([])
        assert summary["total_cases"] == 0

    def test_f1_average(self):
        runner = EvalRunner()
        results = [self._make_result(f1=1.0), self._make_result(f1=0.5)]
        summary = runner.compute_summary(results)
        assert summary["total_cases"] == 2
        assert summary["L1"]["tool_selection_f1_avg"] == 0.75

    def test_hallucination_gate(self):
        runner = EvalRunner()
        results = [self._make_result(hallucination=0.0), self._make_result(hallucination=0.5)]
        summary = runner.compute_summary(results)
        assert summary["action_quality_gate"]["tool_hallucination_ok"] is False

    def test_high_risk_gate(self):
        runner = EvalRunner()
        results = [self._make_result(unconfirmed=0.0), self._make_result(unconfirmed=0.5)]
        summary = runner.compute_summary(results)
        assert summary["action_quality_gate"]["high_risk_all_confirmed"] is False

    def test_l3_goal_average(self):
        runner = EvalRunner()
        results = [self._make_result(goal=4.0), self._make_result(goal=5.0)]
        summary = runner.compute_summary(results)
        assert summary["L3"]["goal_achievement_avg"] == 4.5

    def test_l3_goal_normal_excludes_stop_cases(self):
        # 正常用例 goal=[4.0, 5.0]，对抗用例(有expected_stop) goal=1.0
        # goal_achievement_avg 含全部，goal_achievement_normal_avg 只含正常
        runner = EvalRunner()
        results = [
            self._make_result(goal=4.0),
            self._make_result(goal=5.0),
            self._make_result(goal=1.0, expected_stop=True),
        ]
        summary = runner.compute_summary(results)
        assert summary["L3"]["goal_achievement_avg"] == pytest.approx(3.33, abs=0.01)  # (4+5+1)/3
        assert summary["L3"]["goal_achievement_normal_avg"] == pytest.approx(4.5, abs=0.01)  # (4+5)/2

    def test_hitl_averages(self):
        runner = EvalRunner()
        results = [self._make_result(intent=1.0), self._make_result(intent=0.5)]
        summary = runner.compute_summary(results)
        assert summary["L2"]["intent_accuracy_avg"] == 0.75
        assert summary["L2"]["slot_completeness_avg"] == 0.75

    def test_correct_stop_rate_aggregation(self):
        runner = EvalRunner()
        results = [
            {"L1_execution": {}, "L2_safety": {},
             "correct_stop": {"correct_stop_rate": 1.0}},
            {"L1_execution": {}, "L2_safety": {},
             "correct_stop": {"correct_stop_rate": 0.5}},
        ]
        summary = runner.compute_summary(results)
        assert summary["correct_stop"]["correct_stop_rate"] == 0.75
        assert summary["correct_stop"]["total_with_stop_check"] == 2


class TestModelFamily:
    """模型族判定"""

    @pytest.mark.parametrize("name,expected", [
        ("qwen-max-0919", "qwen"),
        ("qwen3.7-plus", "qwen"),
        ("deepseek-chat", "deepseek"),
        ("deepseek-v3", "deepseek"),
        ("kimi-k2.7-code", "moonshot"),
        ("glm-4", "glm"),
        ("unknown-model", "unknown-model"),
    ])
    def test_family_detection(self, name, expected):
        assert _model_family(name) == expected


class TestValidateJudgeConfig:
    """Judge 配置校验"""

    def test_without_judge_returns_disabled(self, monkeypatch):
        result = validate_judge_config(False)
        assert result == (False, None, "")

    def test_missing_judge_config(self, monkeypatch):
        # 无 judge 配置 → 降级
        monkeypatch.setattr("utils.config.judge_api_key", "")
        monkeypatch.setattr("utils.config.judge_base_url", "")
        monkeypatch.setattr("utils.config.judge_model", "")
        monkeypatch.setattr("utils.config.model_api_key", "k")
        monkeypatch.setattr("utils.config.model_base_url", "u")
        monkeypatch.setattr("utils.config.model_name", "qwen-max")
        result = validate_judge_config(True)
        assert result[0] is False

    def test_same_instance_degrades(self, monkeypatch):
        # Judge 与主模型完全一致 → 降级
        monkeypatch.setattr("utils.config.judge_api_key", "k")
        monkeypatch.setattr("utils.config.judge_base_url", "u")
        monkeypatch.setattr("utils.config.judge_model", "qwen-max")
        monkeypatch.setattr("utils.config.model_api_key", "k")
        monkeypatch.setattr("utils.config.model_base_url", "u")
        monkeypatch.setattr("utils.config.model_name", "qwen-max")
        result = validate_judge_config(True)
        assert result[0] is False

    def test_same_family_degrades(self, monkeypatch):
        # Judge 与主模型同族（都是 qwen）→ 降级
        monkeypatch.setattr("utils.config.judge_api_key", "k1")
        monkeypatch.setattr("utils.config.judge_base_url", "u1")
        monkeypatch.setattr("utils.config.judge_model", "qwen-plus")
        monkeypatch.setattr("utils.config.model_api_key", "k2")
        monkeypatch.setattr("utils.config.model_base_url", "u2")
        monkeypatch.setattr("utils.config.model_name", "qwen-max")
        result = validate_judge_config(True)
        assert result[0] is False

    def test_different_family_ok(self, monkeypatch):
        # 主模型 qwen，Judge kimi（不同族）→ 启用
        monkeypatch.setattr("utils.config.judge_api_key", "k1")
        monkeypatch.setattr("utils.config.judge_base_url", "u1")
        monkeypatch.setattr("utils.config.judge_model", "kimi-k2.7-code")
        monkeypatch.setattr("utils.config.model_api_key", "k2")
        monkeypatch.setattr("utils.config.model_base_url", "u2")
        monkeypatch.setattr("utils.config.model_name", "qwen-max")
        result = validate_judge_config(True)
        assert result[0] is True
