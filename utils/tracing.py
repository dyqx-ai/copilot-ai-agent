"""LLM 调用采集器 — 为评估体系的 Token 成本与延迟统计提供数据。

业务代码（models/llm.py）在每次 LLM 调用成功后调用 record_call()，
仅当评估开启录制（start_recording）时才累积记录，生产环境零开销。

用法（run_eval.py）:
    from utils.tracing import start_recording, stop_recording
    start_recording()
    task = execute_case(...)          # 内部会触发多条 LLM 调用
    llm_logs = stop_recording()       # 返回 [LLMCallRecord, ...]
    trace = collect_trace_from_task(task, llm_call_logs=llm_logs, ...)
"""

import threading

from eval.eval_trace import LLMCallRecord

# 全局记录列表与锁：execute_case 在后台线程执行 LLM 调用，
# 因此必须用全局状态（而非 thread-local）跨线程累积。
_records: list = []
_lock = threading.Lock()
_recording = False


def start_recording():
    """开启录制并清空历史记录，返回 None。"""
    global _recording, _records
    with _lock:
        _recording = True
        _records = []


def stop_recording():
    """关闭录制并返回本次录制的全部记录，同时清空内部缓冲。"""
    global _recording, _records
    with _lock:
        _recording = False
        records, _records = _records, []
    return records


def record_call(model_id, prompt, response, input_tokens=0, output_tokens=0,
                latency_ms=0, step_name="", cache_hit=False):
    """记录一次 LLM 调用。未开启录制时为零操作，保证生产路径无开销。"""
    global _recording
    with _lock:
        if not _recording:
            return
        _records.append(LLMCallRecord(
            step_name=step_name,
            model_id=model_id,
            prompt=prompt,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        ))


def is_recording():
    """当前是否处于录制状态（供测试与调试）。"""
    with _lock:
        return _recording
