from unittest.mock import MagicMock, patch

from empty_space.llm import GeminiClient, GeminiResponse


def _mock_response(text: str = "hello", tokens_in: int = 100, tokens_out: int = 42):
    r = MagicMock()
    r.text = text
    r.usage_metadata.prompt_token_count = tokens_in
    r.usage_metadata.candidates_token_count = tokens_out
    return r


def test_gemini_client_generate_flash_default():
    with patch("empty_space.llm.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response(
            text="主回應內容\n\n---IMPRESSIONS---\n- text: 測試\n  symbols: [a]",
            tokens_in=250,
            tokens_out=42,
        )

        client = GeminiClient(api_key="test_key")
        result = client.generate(
            system="系統提示",
            user="用戶訊息",
        )  # default model

    assert isinstance(result, GeminiResponse)
    assert result.tokens_in == 250
    assert result.tokens_out == 42
    assert result.model == "gemini-2.5-flash"
    assert result.latency_ms >= 0


def test_gemini_client_generate_pro_for_composer():
    with patch("empty_space.llm.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response(
            text="composer snapshot",
            tokens_in=9800,
            tokens_out=2500,
        )

        client = GeminiClient(api_key="test_key")
        result = client.generate(
            system="composer instruction",
            user="貫通軸 + 關係層 + ledger top-15",
            model="gemini-2.5-pro",
        )

    assert result.model == "gemini-2.5-pro"
    assert result.tokens_in == 9800
    assert result.tokens_out == 2500
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-pro"


def test_gemini_client_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env_key")
    with patch("empty_space.llm.genai.Client") as mock_client_cls:
        GeminiClient()
        mock_client_cls.assert_called_once_with(api_key="env_key")
