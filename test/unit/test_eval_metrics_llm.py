"""测试 LLM 采集链路：llm_calls 数据 → Token 成本与 E2E 延迟指标。

验证 collect_trace_from_task 传入 llm_call_logs 后，
calc_token_cost 和 calc_e2e_latency 能产出有意义的数值。
"""

import pytest


def _make_llm_record(model_id, input_tokens, output_tokens, latency_ms):
    from eval.eval_trace import LLMCallRecord
    return LLMCallRecord(
        step_name="step",
        model_id=model_id,
        prompt="prompt",
        response="response",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


class TestTokenCostWithLlmCalls:
    """Token 成本指标在 llm_calls 有数据时能正确计算"""

    def test_token_cost_accumulates(self):
        from eval.eval_metrics import calc_token_cost
        from eval.eval_trace import AgentTrace

        trace = AgentTrace(run_id="r", session_id="s", test_case_id="t",
                           task_input="q", task_output="o")
        trace.llm_calls = [
            _make_llm_record("qwen-max", 100, 50, 1000),
            _make_llm_record("qwen-max", 200, 80, 2000),
        ]

        cost = calc_token_cost(trace)
        assert cost["total_input_tokens"] == 300
        assert cost["total_output_tokens"] == 130
        assert cost["total_tokens"] == 430
        # qwen-max 在默认价目表中，llm_cost 应 > 0
        assert cost["llm_cost_usd"] > 0
        assert cost["tool_cost_usd"] == 0.0
        assert cost["total_cost_usd"] == cost["llm_cost_usd"]

    def test_token_cost_unknown_model_fallback(self):
        from eval.eval_metrics import calc_token_cost
        from eval.eval_trace import AgentTrace

        trace = AgentTrace(run_id="r", session_id="s", test_case_id="t",
                           task_input="q", task_output="o")
        trace.llm_calls = [_make_llm_record("unknown-model", 1000, 1000, 0)]
        cost = calc_token_cost(trace)
        # 未知模型用 fallback 价目表 {0.001, 0.003}
        assert cost["total_tokens"] == 2000
        assert cost["llm_cost_usd"] > 0

    def test_token_cost_empty_llm_calls(self):
        from eval.eval_metrics import calc_token_cost
        from eval.eval_trace import AgentTrace

        trace = AgentTrace(run_id="r", session_id="s", test_case_id="t",
                           task_input="q", task_output="o")
        cost = calc_token_cost(trace)
        assert cost["total_tokens"] == 0
        assert cost["llm_cost_usd"] == 0.0


class TestE2ELatencyWithLlmCalls:
    """E2E 延迟指标在 llm_calls 有数据时能算出 LLM 延迟分位"""

    def test_llm_latency_percentiles(self):
        from eval.eval_metrics import calc_e2e_latency
        from eval.eval_trace import AgentTrace

        trace = AgentTrace(run_id="r", session_id="s", test_case_id="t",
                           task_input="q", task_output="o")
        trace.start_time_ms = 0
        trace.end_time_ms = 10000
        trace.llm_calls = [
            _make_llm_record("m", 0, 0, 1000),
            _make_llm_record("m", 0, 0, 2000),
            _make_llm_record("m", 0, 0, 3000),
        ]

        latency = calc_e2e_latency(trace)
        assert latency["e2e_latency_ms"] == 10000
        assert latency["total_llm_calls"] == 3
        # P95 插值：k=1.9 → 2000*0.1 + 3000*0.9 = 2900（浮点近似）
        assert latency["llm_p50_ms"] == pytest.approx(2000, abs=0.01)
        assert latency["llm_p95_ms"] == pytest.approx(2900, abs=1)


class TestCollectTraceWithLlmLogs:
    """collect_trace_from_task 接收 llm_call_logs 并合并到 trace.llm_calls"""

    def test_llm_logs_merged_into_trace(self):
        from eval.eval_trace import AgentTrace, collect_trace_from_task
        import json

        # 构造一个最小 Task 桩（MongoEngine 文档的属性访问兼容）
        class FakeTask:
            task_id = "task1"
            raw_query = "查询产品"
            status = -1
            system_output = "已完成"
            nodes = []          # 空 nodes，走 curr_tool_id 回退
            curr_tool_id = 0    # 无工具调用
            curr_tool_param = None
            curr_task_desc = ""
            feedback_log = []

        task = FakeTask()
        llm_logs = [_make_llm_record("qwen-max", 10, 5, 100)]
        trace = collect_trace_from_task(task, llm_call_logs=llm_logs)

        assert isinstance(trace, AgentTrace)
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].model_id == "qwen-max"
        assert trace.llm_calls[0].input_tokens == 10

    def test_llm_logs_none_keeps_empty(self):
        from eval.eval_trace import collect_trace_from_task

        class FakeTask:
            task_id = "task1"
            raw_query = "查询产品"
            status = -1
            system_output = "已完成"
            nodes = []
            curr_tool_id = 0
            curr_tool_param = None
            curr_task_desc = ""
            feedback_log = []

        trace = collect_trace_from_task(FakeTask(), llm_call_logs=None)
        assert trace.llm_calls == []
