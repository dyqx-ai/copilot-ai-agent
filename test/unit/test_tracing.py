"""测试 LLM 调用采集器（utils/tracing.py）"""

import pytest


class TestTracing:
    """LLM 调用采集器的核心逻辑"""

    def setup_method(self):
        from utils import tracing
        # 每个测试前确保录制状态复位
        tracing.stop_recording()

    def test_start_then_record_then_stop(self):
        from utils import tracing
        tracing.start_recording()
        tracing.record_call("qwen-max", "prompt1", "resp1", input_tokens=10, output_tokens=5, latency_ms=100)
        tracing.record_call("qwen-max", "prompt2", "resp2", input_tokens=20, output_tokens=8, latency_ms=200)
        records = tracing.stop_recording()

        assert len(records) == 2
        assert records[0].model_id == "qwen-max"
        assert records[0].prompt == "prompt1"
        assert records[0].response == "resp1"
        assert records[0].input_tokens == 10
        assert records[0].output_tokens == 5
        assert records[0].latency_ms == 100
        assert records[1].input_tokens == 20

    def test_no_recording_is_noop(self):
        """未开启录制时 record_call 是零操作，不累积记录"""
        from utils import tracing
        tracing.record_call("qwen-max", "p", "r", input_tokens=1)
        records = tracing.stop_recording()
        assert records == []

    def test_stop_clears_buffer(self):
        """连续两次 start/stop 之间记录不串扰（多用例评估场景）"""
        from utils import tracing
        tracing.start_recording()
        tracing.record_call("m", "p1", "r1", input_tokens=1)
        first = tracing.stop_recording()
        assert len(first) == 1

        tracing.start_recording()
        tracing.record_call("m", "p2", "r2", input_tokens=2)
        second = tracing.stop_recording()
        assert len(second) == 1
        assert second[0].prompt == "p2"  # 不包含第一轮的记录

    def test_is_recording_flag(self):
        from utils import tracing
        assert tracing.is_recording() is False
        tracing.start_recording()
        assert tracing.is_recording() is True
        tracing.stop_recording()
        assert tracing.is_recording() is False

    def test_default_field_values(self):
        """只传必填字段时，可选字段用默认值"""
        from utils import tracing
        tracing.start_recording()
        tracing.record_call("m", "p", "r")
        records = tracing.stop_recording()
        assert records[0].input_tokens == 0
        assert records[0].output_tokens == 0
        assert records[0].latency_ms == 0
        assert records[0].cache_hit is False
        assert records[0].step_name == ""
