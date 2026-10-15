"""测试 eval_trace.py — Trace 数据采集逻辑。

验证 collect_trace_from_task 从 Task 实体正确采集:
- nodes 解析工具调用
- name_mapping 名称映射
- params 回退（nodes 参数被消费后回退 curr_tool_param）
- nodes 为空时 curr_tool_id 回退
- stop_reason 关键词检测
- feedback_log 采集
"""

import json

from eval.eval_trace import AgentTrace, ToolCallRecord, collect_trace_from_task


class FakeTask:
    """模拟 MongoEngine Task 文档的最小桩"""

    def __init__(self, task_id="t1", status=-1, system_output="已完成",
                 raw_query="查询产品", nodes=None, curr_tool_id=0,
                 curr_tool_param=None, curr_task_desc="", feedback_log=None):
        self.task_id = task_id
        self.status = status
        self.system_output = system_output
        self.raw_query = raw_query
        self.nodes = nodes
        self.curr_tool_id = curr_tool_id
        self.curr_tool_param = curr_tool_param
        self.curr_task_desc = curr_task_desc
        self.feedback_log = feedback_log


class TestCollectBasic:
    """基础采集：状态判断、字段填充"""

    def test_completed_task(self):
        task = FakeTask(status=-1, system_output="已完成")
        trace = collect_trace_from_task(task)
        assert isinstance(trace, AgentTrace)
        assert trace.final_status == "completed"
        assert trace.task_success is True
        assert trace.run_id == "t1"
        assert trace.task_input == "查询产品"
        assert trace.task_output == "已完成"

    def test_waiting_confirmation(self):
        task = FakeTask(status=100, system_output="等待确认")
        trace = collect_trace_from_task(task)
        assert trace.final_status == "waiting_confirmation"
        assert trace.task_success is False

    def test_failed_status(self):
        task = FakeTask(status=500, system_output="出错")
        trace = collect_trace_from_task(task)
        assert trace.final_status == "failed"

    def test_cancelled_by_stop_message(self):
        task = FakeTask(status=-1, system_output="任务停止：检测到循环调用")
        trace = collect_trace_from_task(task)
        assert trace.final_status == "cancelled"
        assert trace.task_success is False  # 任务停止 → 不算成功


class TestStopReasonDetection:
    """stop_reason 关键词检测"""

    def test_param_invalid(self):
        task = FakeTask(system_output="参数校验不通过，共2项：...")
        assert collect_trace_from_task(task).stop_reason == "param_invalid"

    def test_param_invalid_keyword(self):
        task = FakeTask(system_output="非法参数")
        assert collect_trace_from_task(task).stop_reason == "param_invalid"

    def test_param_incomplete(self):
        task = FakeTask(system_output="无法为 工具 API 补全缺少参数 id")
        assert collect_trace_from_task(task).stop_reason == "param_incomplete"

    def test_param_incomplete_fully(self):
        task = FakeTask(system_output="无法为 工具 API 完全缺少参数 id")
        assert collect_trace_from_task(task).stop_reason == "param_incomplete"

    def test_tool_not_found(self):
        task = FakeTask(system_output="未找到合适的工具")
        assert collect_trace_from_task(task).stop_reason == "tool_not_found"

    def test_injection_blocked(self):
        task = FakeTask(system_output="检测到提示注入攻击")
        assert collect_trace_from_task(task).stop_reason == "injection_blocked"

    def test_no_stop_reason(self):
        task = FakeTask(system_output="正常完成")
        assert collect_trace_from_task(task).stop_reason is None


class TestCollectFromNodes:
    """从 task.nodes 采集工具调用"""

    def test_single_node(self):
        task = FakeTask(nodes=[
            {"label": "根据产品ID获取产品信息", "params": '{"productId": 1}',
             "task_description": "查询产品", "result": '{"name": "苹果"}'}
        ])
        trace = collect_trace_from_task(task)
        assert len(trace.tool_calls) == 1
        call = trace.tool_calls[0]
        assert call.tool_name == "根据产品ID获取产品信息"
        assert call.params == {"productId": 1}
        assert call.step_name == "查询产品"
        assert call.confirmed_by_user is True  # 能写入 node 说明已确认

    def test_params_dict_direct(self):
        # params 已是 dict 而非 JSON 字符串
        task = FakeTask(nodes=[
            {"label": "tool1", "params": {"id": 5}, "task_description": "", "result": ""}
        ])
        trace = collect_trace_from_task(task)
        assert trace.tool_calls[0].params == {"id": 5}

    def test_params_invalid_json_falls_back_empty(self):
        task = FakeTask(nodes=[
            {"label": "tool1", "params": "{invalid json", "task_description": "", "result": ""}
        ])
        trace = collect_trace_from_task(task)
        assert trace.tool_calls[0].params == {}

    def test_error_label_detection(self):
        # label 含"异常" → 记为 error
        task = FakeTask(nodes=[
            {"label": "异常调用节点-缺少必要参数", "params": "{}", "task_description": "", "result": ""}
        ])
        trace = collect_trace_from_task(task)
        assert trace.tool_calls[0].error is not None
        assert "异常" in trace.tool_calls[0].error

    def test_name_mapping_applied(self):
        # name_mapping 将中文名映射为 name_for_model
        task = FakeTask(nodes=[
            {"label": "根据产品ID获取产品信息", "params": "{}", "task_description": "", "result": ""}
        ])
        name_mapping = {"根据产品ID获取产品信息": "tool10"}
        trace = collect_trace_from_task(task, name_mapping=name_mapping)
        assert trace.tool_calls[0].tool_name == "tool10"

    def test_params_fallback_to_curr_tool_param(self):
        # nodes 的 params 为空（被 tool_use 消费），回退到 curr_tool_param
        task = FakeTask(
            nodes=[{"label": "tool1", "params": "{}", "task_description": "", "result": ""}],
            curr_tool_param={"id": 99},
        )
        trace = collect_trace_from_task(task)
        assert trace.tool_calls[0].params == {"id": 99}

    def test_result_summary_truncated(self):
        task = FakeTask(nodes=[
            {"label": "tool1", "params": "{}", "task_description": "",
             "result": "x" * 500}
        ])
        trace = collect_trace_from_task(task)
        assert len(trace.tool_calls[0].result_summary) <= 200


class TestCollectFromCurrTool:
    """nodes 为空时回退到 curr_tool_id + curr_tool_param"""

    def test_curr_tool_id_fallback(self):
        task = FakeTask(nodes=[], curr_tool_id=5, curr_tool_param={"id": 1},
                        curr_task_desc="查产品")
        trace = collect_trace_from_task(task)
        assert len(trace.tool_calls) == 1
        call = trace.tool_calls[0]
        assert call.tool_name == "tool5"  # 无映射时回退 tool{id}
        assert call.params == {"id": 1}
        assert call.step_name == "查产品"

    def test_curr_tool_id_with_tool_id_map(self):
        # tool_id_map 将 Milvus 内部 ID 映射为 name_for_model
        task = FakeTask(nodes=[], curr_tool_id=52, curr_tool_param={})
        tool_id_map = {52: "tool10"}
        trace = collect_trace_from_task(task, tool_id_map=tool_id_map)
        assert trace.tool_calls[0].tool_name == "tool10"

    def test_injection_rejected_error(self):
        task = FakeTask(nodes=[], curr_tool_id=1, system_output="检测到提示注入攻击，已拒绝执行")
        trace = collect_trace_from_task(task)
        assert "注入攻击" in trace.tool_calls[0].error

    def test_param_rejected_error(self):
        task = FakeTask(nodes=[], curr_tool_id=1, system_output="参数校验不通过")
        trace = collect_trace_from_task(task)
        assert trace.tool_calls[0].error is not None

    def test_stopped_missing_param_error(self):
        task = FakeTask(nodes=[], curr_tool_id=1, status=-1,
                        system_output="任务停止：缺少必要参数")
        trace = collect_trace_from_task(task)
        assert trace.tool_calls[0].error is not None

    def test_no_tool_no_fallback(self):
        # 无 nodes、无 curr_tool_id → 无工具调用
        task = FakeTask(nodes=[], curr_tool_id=0)
        trace = collect_trace_from_task(task)
        assert trace.tool_calls == []

    def test_curr_tool_confirmed_status(self):
        # 回退路径 confirmed_by_user = (task_status == -1)
        task = FakeTask(nodes=[], curr_tool_id=1, status=-1)
        trace = collect_trace_from_task(task)
        assert trace.tool_calls[0].confirmed_by_user is True

        task2 = FakeTask(nodes=[], curr_tool_id=1, status=100)
        trace2 = collect_trace_from_task(task2)
        assert trace2.tool_calls[0].confirmed_by_user is False


class TestFeedbackLogCollection:
    """HITL 反馈审计日志采集"""

    def test_feedback_log_collected(self):
        feedback = [{"intent": "confirm", "patch": {}}, {"intent": "abort"}]
        task = FakeTask(feedback_log=feedback)
        trace = collect_trace_from_task(task)
        assert trace.feedback_log == feedback

    def test_feedback_log_empty(self):
        task = FakeTask(feedback_log=None)
        trace = collect_trace_from_task(task)
        assert trace.feedback_log == []


class TestLlmCallsMerge:
    """LLM 调用日志合并"""

    def test_llm_logs_merged(self):
        from eval.eval_trace import LLMCallRecord
        task = FakeTask()
        llm_logs = [LLMCallRecord(step_name="s", model_id="qwen-max",
                                  prompt="p", response="r", input_tokens=10)]
        trace = collect_trace_from_task(task, llm_call_logs=llm_logs)
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].input_tokens == 10

    def test_llm_logs_none(self):
        task = FakeTask()
        trace = collect_trace_from_task(task, llm_call_logs=None)
        assert trace.llm_calls == []


class TestExtractToolNames:
    """extract_tool_names_from_task — 路径稳定性辅助"""

    def test_extract_valid_labels(self):
        task = FakeTask(nodes=[
            {"label": "tool1", "params": "{}", "task_description": "", "result": ""},
            {"label": "tool2", "params": "{}", "task_description": "", "result": ""},
        ])
        from eval.eval_trace import extract_tool_names_from_task
        assert extract_tool_names_from_task(task) == ["tool1", "tool2"]

    def test_skip_error_labels(self):
        task = FakeTask(nodes=[
            {"label": "tool1", "params": "{}", "task_description": "", "result": ""},
            {"label": "异常调用节点", "params": "{}", "task_description": "", "result": ""},
        ])
        from eval.eval_trace import extract_tool_names_from_task
        assert extract_tool_names_from_task(task) == ["tool1"]

    def test_empty_nodes(self):
        from eval.eval_trace import extract_tool_names_from_task
        assert extract_tool_names_from_task(FakeTask(nodes=None)) == []
