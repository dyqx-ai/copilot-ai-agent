"""评估指标函数的坏数据韧性测试。

核心目的：保证评估代码在任何异常/空/畸形输入下都能正常运行而不崩溃。
这是"评估代码本身无 bug"的第一道防线——喂空 trace、畸形数据必须返回合理兜底值。
"""

import pytest

from eval.eval_trace import AgentTrace, ToolCallRecord, LLMCallRecord
from eval import eval_metrics as M


def _empty_trace():
    """构造一个全空的 AgentTrace"""
    return AgentTrace(run_id="", session_id="", test_case_id="",
                      task_input="", task_output="")


def _make_tool_call(tool_name="tool1", params=None, error=None):
    """构造工具调用记录"""
    return ToolCallRecord(step_name="step", tool_name=tool_name,
                          params=params or {}, error=error)


class TestMetricsNeverCrashOnEmpty:
    """喂空数据时所有指标必须不崩溃并返回 dict"""

    def test_tool_selection_f1_empty(self):
        # 空调用 + 空期望 → TP=FP=FN=0，Precision/Recall 走 else 1.0 → F1=1.0
        # 语义：没有期望工具也没调用工具 = 完美匹配（对抗用例 expected_tools=[] 且系统未调用工具时）
        result = M.calc_tool_selection_precision_recall([], [])
        assert isinstance(result, dict)
        assert result["f1"] == 1.0

    def test_tool_selection_f1_no_expected(self):
        # expected_tools 为空时，实际有调用也不该崩溃
        result = M.calc_tool_selection_precision_recall([_make_tool_call("tool1")], [])
        assert isinstance(result, dict)

    def test_tool_hallucination_empty_calls(self):
        result = M.calc_tool_hallucination_rate([], set())
        assert result["hallucination_rate"] == 0.0

    def test_tool_hallucination_empty_available(self):
        # available_tools 为空时，任何调用都应视为幻觉但不崩溃
        result = M.calc_tool_hallucination_rate([_make_tool_call("tool1")], set())
        assert isinstance(result, dict)
        assert result["hallucination_rate"] == 1.0

    def test_param_existence_empty(self):
        result = M.calc_param_existence([], {})
        assert result["existence_rate"] == 1.0

    def test_param_existence_unknown_tool(self):
        # tool_schemas 里没有该工具的 schema，跳过不崩溃
        result = M.calc_param_existence([_make_tool_call("ghost_tool", {"a": 1})], {})
        assert result["existence_rate"] == 1.0

    def test_param_type_empty(self):
        result = M.calc_param_type_correctness([], {})
        assert result["type_correctness_rate"] == 1.0

    def test_tool_call_failure_empty(self):
        result = M.calc_tool_call_failure_rate([])
        assert result["failure_rate"] == 0.0

    def test_call_redundancy_empty(self):
        result = M.calc_call_redundancy([])
        assert result["redundancy_rate"] == 0.0

    def test_token_cost_empty(self):
        result = M.calc_token_cost(_empty_trace())
        assert result["total_tokens"] == 0
        assert result["total_cost_usd"] == 0.0

    def test_e2e_latency_empty(self):
        result = M.calc_e2e_latency(_empty_trace())
        assert result["e2e_latency_ms"] == 0
        assert result["total_llm_calls"] == 0
        assert result["total_tool_calls"] == 0

    def test_high_risk_empty(self):
        result = M.calc_high_risk_action_without_confirmation([])
        assert result["missing_confirmation_rate"] == 0.0

    def test_permission_violation_empty_trace(self):
        result = M.calc_permission_violation_rate(_empty_trace(), allowed_tools=set())
        assert result["permission_violation_rate"] == 0.0

    def test_permission_violation_none_allowed(self):
        # allowed_tools=None 表示不限制，跳过检查
        result = M.calc_permission_violation_rate(_empty_trace(), allowed_tools=None)
        assert result["permission_violation_rate"] == 0.0
        assert result["applicable"] is False

    def test_audit_trail_empty(self):
        result = M.calc_audit_trail_completeness(_empty_trace())
        assert result["audit_completeness"] == 1.0

    def test_correct_stop_empty(self):
        result = M.calc_correct_stop_rate([], "param_incomplete")
        assert result["correct_stop_rate"] == 1.0
        assert result["total"] == 0

    def test_output_consistency_empty(self):
        result = M.calc_output_consistency([])
        assert result["cv"] == 0.0

    def test_output_consistency_single(self):
        # 只有 1 个分数时，CV 应为 0
        result = M.calc_output_consistency([4.0])
        assert result["cv"] == 0.0
        assert result["n_runs"] == 1

    def test_path_stability_empty(self):
        result = M.calc_path_stability([])
        assert result["dominant_path_rate"] == 0.0
        assert result["n_runs"] == 0

    def test_intent_accuracy_no_expected(self):
        result = M.calc_intent_accuracy([], [])
        assert result["applicable"] is False
        assert result["intent_accuracy"] == 1.0

    def test_slot_filling_no_expected(self):
        result = M.calc_slot_filling_completeness([], [])
        assert result["applicable"] is False
        assert result["slot_completeness"] == 1.0

    def test_task_success_none_score(self):
        # L3 未启用时传入 None，不应崩溃
        result = M.calc_task_success(None)
        assert result["task_success"] is None
        assert "reason" in result


class TestMetricsNeverCrashOnMalformed:
    """喂畸形数据时也必须不崩溃"""

    def test_tool_call_with_missing_fields(self):
        # params 为 None、error 为 None 的畸形调用
        call = ToolCallRecord(step_name="", tool_name="", params=None, error=None)
        result = M.calc_param_existence([call], {})
        assert isinstance(result, dict)

    def test_param_value_none(self):
        # 参数值为 None
        call = _make_tool_call("tool1", {"quantity": None})
        schemas = {"tool1": {"required": ["quantity"], "properties": {"quantity": {"type": "int"}}}}
        result = M.calc_param_existence([call], schemas)
        assert result["total_missing"] == 1  # None 视为缺失

    def test_hallucination_mixed_valid_invalid(self):
        calls = [_make_tool_call("tool1"), _make_tool_call("ghost_tool")]
        result = M.calc_tool_hallucination_rate(calls, {"tool1"})
        assert result["hallucination_rate"] == 0.5

    def test_redundancy_with_none_params(self):
        # 两个参数为 None 的调用，不应在序列化时崩溃
        c1 = ToolCallRecord(step_name="", tool_name="t", params=None)
        c2 = ToolCallRecord(step_name="", tool_name="t", params=None)
        result = M.calc_call_redundancy([c1, c2])
        assert isinstance(result, dict)


class TestL3MetricsWithNoneLlm:
    """L3 指标在 llm=None（--with-judge 未启用）时必须兜底不崩溃"""

    @pytest.mark.parametrize("name", [
        "calc_goal_achievement",
        "calc_factual_accuracy",
        "calc_plan_soundness",
        "calc_evidence_to_action_consistency",
        "calc_error_diagnosis_quality",
        "calc_param_semantic_correctness",
        "calc_statement_hallucination_rate",
        "calc_overconfidence",
    ])
    def test_l3_metric_with_none_llm(self, name):
        fn = getattr(M, name)
        trace = _empty_trace()
        result = fn(trace, llm=None, model="")
        assert isinstance(result, dict)


class TestTracingEmptyTrace:
    """坏数据场景下 collect_trace_from_task 不崩溃"""

    def test_task_without_nodes(self):
        from eval.eval_trace import collect_trace_from_task

        class MinimalTask:
            task_id = "t1"
            raw_query = "查询"
            status = -1
            system_output = "已完成"
            nodes = None           # 畸形：nodes 为 None
            curr_tool_id = 0
            curr_tool_param = None
            curr_task_desc = ""
            feedback_log = None

        trace = collect_trace_from_task(MinimalTask())
        assert isinstance(trace, AgentTrace)

    def test_task_nodes_is_none(self):
        from eval.eval_trace import collect_trace_from_task

        class TaskNoNodes:
            task_id = "t2"
            raw_query = "查询"
            status = 100
            system_output = "等待确认"
            nodes = []
            curr_tool_id = 5
            curr_tool_param = {"id": 1}
            curr_task_desc = "查产品"
            feedback_log = []

        trace = collect_trace_from_task(TaskNoNodes())
        # 应走 curr_tool_id 回退，有 1 条工具调用
        assert isinstance(trace, AgentTrace)
