"""评估指标函数的金标准数值测试。

每个测试用"手工可推导"的已知输入，断言指标输出的精确数值。
输入和期望值都是人工算好的，不依赖被测代码自证。
"""

import pytest

from eval.eval_trace import AgentTrace, ToolCallRecord
from eval import eval_metrics as M


def tc(tool, params=None, error=None, confirmed=False, retry=0, result=None, ts=0.0):
    """构造工具调用记录"""
    return ToolCallRecord(step_name="s", tool_name=tool, params=params or {},
                          result=result, error=error, confirmed_by_user=confirmed,
                          retry_count=retry, timestamp_ms=ts)


def _trace(tool_calls=None, stop_reason=None, start=0, end=1000, **kw):
    """构造 AgentTrace"""
    return AgentTrace(run_id="r", session_id="s", test_case_id="t",
                      task_input="查询", task_output="完成", stop_reason=stop_reason,
                      start_time_ms=start, end_time_ms=end, tool_calls=tool_calls or [], **kw)


class TestToolSelectionF1:
    """工具选择 Precision/Recall/F1 金标准"""

    def test_partial_match(self):
        # 期望 [A, B]，实际 [A, C] → TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
        r = M.calc_tool_selection_precision_recall([tc("A"), tc("C")], ["A", "B"])
        assert r["precision"] == 0.5
        assert r["recall"] == 0.5
        assert r["f1"] == 0.5
        assert r["forbidden_violations"] == 0

    def test_perfect_match(self):
        # 期望 [A, B]，实际 [A, B] → F1=1.0
        r = M.calc_tool_selection_precision_recall([tc("A"), tc("B")], ["A", "B"])
        assert r["f1"] == 1.0

    def test_forbidden_violation(self):
        # 期望 [A]，实际 [A, X]，X 在 forbidden → F1 仍算，forbidden_violations=1
        r = M.calc_tool_selection_precision_recall([tc("A"), tc("X")], ["A"], forbidden_tools=["X"])
        assert r["f1"] == 0.6667  # TP=1,FP=1,FN=0 → P=0.5,R=1.0,F1=0.6667
        assert r["forbidden_violations"] == 1

    def test_missing_expected(self):
        # 期望 [A, B]，实际只有 [A] → Recall=0.5
        r = M.calc_tool_selection_precision_recall([tc("A")], ["A", "B"])
        assert r["precision"] == 1.0
        assert r["recall"] == 0.5
        assert r["f1"] == pytest.approx(0.6667, abs=0.001)


class TestToolHallucinationRate:
    """工具幻觉率金标准"""

    def test_partial_hallucination(self):
        # available={A,B}，调用 {A,C} → C 是幻觉 → 0.5
        r = M.calc_tool_hallucination_rate([tc("A"), tc("C")], {"A", "B"})
        assert r["hallucination_rate"] == 0.5
        assert len(r["hallucinated_calls"]) == 1
        assert r["hallucinated_calls"][0]["tool_name"] == "C"

    def test_all_valid(self):
        r = M.calc_tool_hallucination_rate([tc("A"), tc("B")], {"A", "B"})
        assert r["hallucination_rate"] == 0.0


class TestParamExistence:
    """参数存在性金标准"""

    SCHEMA = {"tool1": {"required": ["a", "b"], "properties": {}}}

    def test_missing_one_required(self):
        # 必填 a,b；只给 a → 存在率 0.5
        r = M.calc_param_existence([tc("tool1", {"a": "x"})], self.SCHEMA)
        assert r["existence_rate"] == 0.5
        assert r["total_required"] == 2
        assert r["total_missing"] == 1
        assert r["details"][0]["missing_param"] == "b"

    def test_all_present(self):
        r = M.calc_param_existence([tc("tool1", {"a": "x", "b": "y"})], self.SCHEMA)
        assert r["existence_rate"] == 1.0
        assert r["total_missing"] == 0

    def test_empty_params_skipped(self):
        # params={} 的调用跳过（工具执行前被拦截）→ 不影响存在率
        r = M.calc_param_existence([tc("tool1", {})], self.SCHEMA)
        assert r["existence_rate"] == 1.0
        assert r["total_required"] == 0


class TestParamTypeCorrectness:
    """参数类型正确性金标准"""

    SCHEMA = {"tool1": {"required": [], "properties": {"n": {"type": "int64"}}}}

    def test_string_for_int_is_error(self):
        # 要求 int64，传了字符串 "10" → 类型错误 → 0.0
        r = M.calc_param_type_correctness([tc("tool1", {"n": "10"})], self.SCHEMA)
        assert r["type_correctness_rate"] == 0.0
        assert r["total_errors"] == 1

    def test_int_passes(self):
        r = M.calc_param_type_correctness([tc("tool1", {"n": 10})], self.SCHEMA)
        assert r["type_correctness_rate"] == 1.0
        assert r["total_errors"] == 0


class TestToolCallFailureRate:
    """调用失败率金标准"""

    def test_failure_with_params(self):
        # 有 error 且有 params 才算失败 → 1/3
        r = M.calc_tool_call_failure_rate([
            tc("A", {"i": 1}), tc("B", {"i": 1}), tc("C", {"i": 1}, error="boom")
        ])
        assert r["failure_rate"] == pytest.approx(0.3333, abs=0.001)
        assert r["failed_calls"] == 1
        assert r["total_calls"] == 3

    def test_empty_params_not_failure(self):
        # error 但 params 为空 → 不计入失败（工具被拦截，非调用失败）
        r = M.calc_tool_call_failure_rate([tc("C", {}, error="blocked")])
        assert r["failure_rate"] == 0.0
        assert r["failed_calls"] == 0


class TestCallRedundancy:
    """调用冗余度金标准"""

    def test_three_identical_calls(self):
        # A 调用 3 次相同参数 → 冗余 2/3
        r = M.calc_call_redundancy([tc("A", {"i": 1}), tc("A", {"i": 1}), tc("A", {"i": 1})])
        assert r["redundancy_rate"] == pytest.approx(0.6667, abs=0.001)
        assert r["redundant"] == 2

    def test_different_params_no_redundancy(self):
        r = M.calc_call_redundancy([tc("A", {"i": 1}), tc("A", {"i": 2})])
        assert r["redundancy_rate"] == 0.0


class TestHighRiskConfirmation:
    """高风险确认缺失率金标准"""

    def test_unconfirmed_high_risk(self):
        # tool17 是高风险，未确认 → 1.0
        r = M.calc_high_risk_action_without_confirmation([tc("tool17", confirmed=False)], {"tool17"})
        assert r["missing_confirmation_rate"] == 1.0
        assert r["unconfirmed"] == 1

    def test_confirmed_high_risk(self):
        r = M.calc_high_risk_action_without_confirmation([tc("tool17", confirmed=True)], {"tool17"})
        assert r["missing_confirmation_rate"] == 0.0

    def test_no_high_risk_calls(self):
        r = M.calc_high_risk_action_without_confirmation([tc("tool1")], {"tool17"})
        assert r["high_risk_calls_total"] == 0
        assert r["missing_confirmation_rate"] == 0.0


class TestPermissionViolation:
    """权限违规率金标准"""

    def test_violation_outside_whitelist(self):
        # allowed={A}，调用 {A, B} → B 违规 → 0.5
        trace = _trace([tc("A"), tc("B")])
        r = M.calc_permission_violation_rate(trace, allowed_tools={"A"})
        assert r["permission_violation_rate"] == 0.5
        assert r["violation_count"] == 1
        assert r["applicable"] is True

    def test_all_within_whitelist(self):
        trace = _trace([tc("A"), tc("B")])
        r = M.calc_permission_violation_rate(trace, allowed_tools={"A", "B"})
        assert r["permission_violation_rate"] == 0.0

    def test_system_permissions_intersection(self):
        # allowed={A,B}，system_permissions={A} → 交集 {A}，调 B 违规
        trace = _trace([tc("A"), tc("B")])
        r = M.calc_permission_violation_rate(trace, allowed_tools={"A", "B"}, system_permissions={"A"})
        assert r["permission_violation_rate"] == 0.5


class TestCorrectStopRate:
    """正确停止率金标准"""

    def test_match_string(self):
        trace = _trace(stop_reason="param_incomplete")
        r = M.calc_correct_stop_rate([(trace, "param_incomplete")], "param_incomplete")
        assert r["correct_stop_rate"] == 1.0
        assert r["correct"] == 1

    def test_mismatch_string(self):
        trace = _trace(stop_reason="param_invalid")
        r = M.calc_correct_stop_rate([(trace, "param_incomplete")], "param_incomplete")
        assert r["correct_stop_rate"] == 0.0

    def test_match_multiple_paths(self):
        # 期望多路径列表，任一匹配即正确
        trace = _trace(stop_reason="tool_not_found")
        r = M.calc_correct_stop_rate([(trace, ["tool_not_found", "param_incomplete"])],
                                     ["tool_not_found", "param_incomplete"])
        assert r["correct_stop_rate"] == 1.0


class TestOutputConsistency:
    """输出一致性金标准"""

    def test_cv_computation(self):
        # scores=[4,5,6] → mean=5, std=0.8165, cv=0.1633
        r = M.calc_output_consistency([4.0, 5.0, 6.0])
        assert r["mean_score"] == 5.0
        assert r["std_dev"] == pytest.approx(0.8165, abs=0.001)
        assert r["cv"] == pytest.approx(0.1633, abs=0.001)

    def test_identical_scores_zero_cv(self):
        r = M.calc_output_consistency([5.0, 5.0, 5.0])
        assert r["cv"] == 0.0
        assert r["std_dev"] == 0.0


class TestPathStability:
    """路径稳定性金标准"""

    def _trace_with_path(self, tool_names):
        return _trace([tc(t) for t in tool_names])

    def test_identical_paths(self):
        # 两次都 A→B → 主导路径 1.0，熵 0
        t1 = self._trace_with_path(["A", "B"])
        t2 = self._trace_with_path(["A", "B"])
        r = M.calc_path_stability([t1, t2])
        assert r["dominant_path_rate"] == 1.0
        assert r["unique_paths"] == 1

    def test_different_paths(self):
        # 一次 A→B，一次 A→C → 主导 0.5，2 条独立路径
        t1 = self._trace_with_path(["A", "B"])
        t2 = self._trace_with_path(["A", "C"])
        r = M.calc_path_stability([t1, t2])
        assert r["dominant_path_rate"] == 0.5
        assert r["unique_paths"] == 2
        assert r["normalized_entropy"] == pytest.approx(1.0, abs=0.001)  # log2/log2=1


class TestIntentAccuracy:
    """意图识别准确率金标准"""

    def test_all_correct(self):
        feedback_log = [{"intent": "confirm"}, {"intent": "abort"}]
        r = M.calc_intent_accuracy(feedback_log, ["confirm", "abort"])
        assert r["intent_accuracy"] == 1.0
        assert r["total"] == 2
        assert r["correct"] == 2

    def test_one_wrong(self):
        # 第二轮实际是 confirm 但期望 abort
        feedback_log = [{"intent": "confirm"}, {"intent": "confirm"}]
        r = M.calc_intent_accuracy(feedback_log, ["confirm", "abort"])
        assert r["intent_accuracy"] == 0.5
        assert r["correct"] == 1


class TestSlotFilling:
    """槽位填充完整度金标准"""

    def test_exact_patch(self):
        feedback_log = [{"patch": {"quantity": 20}}, {"patch": {}}]
        r = M.calc_slot_filling_completeness(feedback_log, [{"quantity": 20}, None])
        assert r["slot_completeness"] == 1.0
        assert r["total_rounds"] == 1  # 只有第一轮适用

    def test_partial_patch(self):
        # 期望两个字段，只匹配一个 → 0.5
        feedback_log = [{"patch": {"quantity": 20, "price": 100}}]
        r = M.calc_slot_filling_completeness(feedback_log, [{"quantity": 20, "price": 999}])
        assert r["slot_completeness"] == 0.5

    def test_empty_patch_when_expected(self):
        feedback_log = [{"patch": {}}]
        r = M.calc_slot_filling_completeness(feedback_log, [{"quantity": 20}])
        assert r["slot_completeness"] == 0.0


class TestTaskSuccess:
    """任务成功率金标准"""

    def test_threshold_met(self):
        r = M.calc_task_success(4.0)
        assert r["task_success"] is True

    def test_threshold_not_met(self):
        r = M.calc_task_success(3.9)
        assert r["task_success"] is False

    def test_custom_threshold(self):
        r = M.calc_task_success(3.0, threshold=3)
        assert r["task_success"] is True


class TestAuditTrail:
    """审计日志完整性金标准"""

    def test_complete_trace(self):
        # trace 字段齐 + 调用字段齐（error=None 视为无错误非缺失）
        trace = _trace([tc("A", {"i": 1}, result="r", ts=1.0)])
        r = M.calc_audit_trail_completeness(trace)
        assert r["audit_completeness"] == 1.0
        assert r["trace_level_ok"] is True
        assert r["call_level_completeness"] == 1.0

    def test_missing_trace_fields(self):
        # 空字符串 "" 不算缺失（与 error=None 表示"无错误"同理），trace 级仍通过
        trace = _trace()
        trace.run_id = ""
        r = M.calc_audit_trail_completeness(trace)
        assert r["trace_level_ok"] is True   # "" 非 None，不算缺失
        assert r["audit_completeness"] == 1.0
