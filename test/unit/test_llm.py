"""测试 LLM 模型包装器"""

import pytest
from unittest.mock import MagicMock, patch


class TestLargeLanguageModel:
    """LLM 包装器"""

    def test_init_creates_openai_client(self):
        with patch("models.llm.OpenAI") as mock_openai:
            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            mock_openai.assert_called_once()
            assert llm.retries == 3

    def test_chat_completions_returns_content(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "test response"
            mock_client.chat.completions.create.return_value = mock_response

            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            result = llm.chat_completions("test prompt", "test-model", 0.01, 0.01)
            assert result == "test response"

    def test_chat_completions_retry_on_failure(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "retry response"

            # 第一次调用失败，第二次成功
            mock_client.chat.completions.create.side_effect = [
                Exception("API error"),
                mock_response,
            ]

            with patch("models.llm.random.randint", return_value=1):
                from models.llm import LargeLanguageModel
                llm = LargeLanguageModel("http://test.api", "test-key")
                result = llm.chat_completions("test prompt", "test-model", 0.01, 0.01)
                assert result == "retry response"
                assert mock_client.chat.completions.create.call_count == 2

    def test_chat_completions_max_retries_exhausted(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.side_effect = Exception("API error")

            with patch("models.llm.random.randint", return_value=1):
                from models.llm import LargeLanguageModel
                llm = LargeLanguageModel("http://test.api", "test-key")
                with pytest.raises(Exception, match="API error"):
                    llm.chat_completions("test prompt", "test-model", 0.01, 0.01)
                assert mock_client.chat.completions.create.call_count == 3

    def test_chat_completions_system_prompt(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "response"
            mock_client.chat.completions.create.return_value = mock_response

            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            llm.chat_completions("test", "model", 0.01, 0.01)

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) >= 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    def test_backoff_range(self):
        with patch("models.llm.OpenAI"):
            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            with patch("models.llm.random.randint", return_value=5) as mock_randint:
                with patch("models.llm.time.sleep") as mock_sleep:
                    llm.backoff()
                    mock_randint.assert_called_once_with(1, 20)
                    mock_sleep.assert_called_once_with(5)

    def test_context_chat_completions(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "context response"
            mock_client.chat.completions.create.return_value = mock_response

            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            contexts = [{"role": "user", "content": "hello"}]
            result = llm.context_chat_completions(contexts, "model", 0.01, 0.01, context_number=5)
            assert result == "context response"

    def test_context_chat_completions_retry(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "retry context response"

            mock_client.chat.completions.create.side_effect = [
                Exception("error"),
                mock_response,
            ]

            with patch("models.llm.random.randint", return_value=1):
                from models.llm import LargeLanguageModel
                llm = LargeLanguageModel("http://test.api", "test-key")
                contexts = [{"role": "user", "content": "hello"}]
                result = llm.context_chat_completions(contexts, "model", 0.01, 0.01, context_number=5)
                assert result == "retry context response"
                assert mock_client.chat.completions.create.call_count == 2


class TestLLMTracing:
    """LLM 埋点采集 — 验证 chat_completions 记录调用，且不破坏现有行为"""

    def setup_method(self):
        from utils import tracing
        tracing.stop_recording()

    def _make_response(self, content="resp", input_tokens=10, output_tokens=20):
        from unittest.mock import MagicMock
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.usage.prompt_tokens = input_tokens
        mock_response.usage.completion_tokens = output_tokens
        return mock_response

    def test_chat_completions_records_call(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_response = self._make_response()
            mock_openai.return_value.chat.completions.create.return_value = mock_response

            from utils.tracing import start_recording, stop_recording
            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            start_recording()
            try:
                result = llm.chat_completions("my prompt", "test-model", 0.01, 0.01)
            finally:
                records = stop_recording()

            assert result == "resp"                    # 返回行为不变
            assert len(records) == 1
            assert records[0].model_id == "test-model"
            assert records[0].prompt == "my prompt"
            assert records[0].response == "resp"
            assert records[0].input_tokens == 10
            assert records[0].output_tokens == 20

    def test_records_latency_ms(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = self._make_response()

            from utils.tracing import start_recording, stop_recording
            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            start_recording()
            try:
                llm.chat_completions("p", "m", 0.01, 0.01)
            finally:
                records = stop_recording()

            # latency 是真实 elapsed，至少 >= 0
            assert records[0].latency_ms >= 0

    def test_no_recording_no_crash(self):
        """未开启录制时，LLM 调用不应因埋点崩溃（生产路径零开销）"""
        with patch("models.llm.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = self._make_response()

            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            # 不调用 start_recording
            result = llm.chat_completions("p", "m", 0.01, 0.01)
            assert result == "resp"

    def test_mock_without_usage_does_not_crash(self):
        """Mock 响应没有 usage 字段时（旧测试场景），埋点应容错"""
        with patch("models.llm.OpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "resp"
            # 不设置 usage 属性
            mock_openai.return_value.chat.completions.create.return_value = mock_response

            from utils.tracing import start_recording, stop_recording
            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            start_recording()
            try:
                result = llm.chat_completions("p", "m", 0.01, 0.01)
            finally:
                records = stop_recording()

            assert result == "resp"                    # 正常返回
            assert len(records) == 1                   # 仍记录了一条
            assert records[0].input_tokens == 0        # usage 缺失时归 0

    def test_context_chat_completions_records(self):
        with patch("models.llm.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = self._make_response(content="ctx-resp")

            from utils.tracing import start_recording, stop_recording
            from models.llm import LargeLanguageModel
            llm = LargeLanguageModel("http://test.api", "test-key")
            contexts = [{"role": "user", "content": "hello"}]
            start_recording()
            try:
                result = llm.context_chat_completions(contexts, "m", 0.01, 0.01, context_number=5)
            finally:
                records = stop_recording()

            assert result == "ctx-resp"
            assert len(records) == 1
            assert records[0].input_tokens == 10
