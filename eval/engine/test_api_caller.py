import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


_MODULE_PATH = Path(__file__).with_name("api_caller.py")
_SPEC = importlib.util.spec_from_file_location("api_caller_under_test", _MODULE_PATH)
_API_CALLER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_API_CALLER)
_add_reasoning_param = _API_CALLER._add_reasoning_param
_parse_api_response = _API_CALLER._parse_api_response


class _Response:
    def __init__(self, body, status_code=200):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class MiniMaxApiCallerTest(unittest.TestCase):
    def test_adds_reasoning_effort_for_minimax_flash(self):
        with patch.dict(
            os.environ,
            {"REASONING_EFFORT": "LOW", "REASONING_THINKING": "adaptive"},
        ):
            payload = {}
            _add_reasoning_param(payload, "MiniMax-M3.1-Flash")

        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertNotIn("thinking", payload)

    def test_omits_unsupported_none_effort_for_minimax_flash(self):
        with patch.dict(os.environ, {"REASONING_EFFORT": "none"}):
            payload = {}
            _add_reasoning_param(payload, "MiniMax-M3.1-Flash")

        self.assertNotIn("reasoning_effort", payload)

    def test_adds_supported_thinking_modes_for_minimax_m3(self):
        for mode in ("adaptive", "disabled"):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"REASONING_THINKING": mode.upper()}
            ):
                payload = {}
                _add_reasoning_param(payload, "MiniMax-M3")

                self.assertEqual(payload["thinking"], {"type": mode})

    def test_does_not_treat_flash_as_minimax_m3(self):
        with patch.dict(os.environ, {"REASONING_THINKING": "disabled"}):
            payload = {}
            _add_reasoning_param(payload, "MiniMax-M3.1-Flash")

        self.assertNotIn("thinking", payload)

    def test_preserves_explicit_legacy_model_without_reasoning_fields(self):
        with patch.dict(
            os.environ,
            {"REASONING_EFFORT": "high", "REASONING_THINKING": "adaptive"},
        ):
            payload = {}
            _add_reasoning_param(payload, "MiniMax-M2.7")

        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("thinking", payload)

    def test_serializes_flash_primary_and_m3_fallback_at_request_boundary(self):
        responses = [
            _Response({"error": {"message": "temporary"}}, status_code=500),
            _Response({"choices": [{"message": {"content": "answer"}}]}),
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            }
        ]
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        sampling_params = {"temperature": 0.2, "top_p": 1.0, "max_tokens": 128}

        with patch.dict(
            os.environ,
            {
                "REASONING_API_KEY": "primary-test-key",
                "REASONING_END_POINT": "https://primary.invalid/v1/chat/completions",
                "REASONING_MODEL_NAME_2": "MiniMax-M3",
                "REASONING_API_KEY_2": "fallback-test-key",
                "REASONING_END_POINT_2": "https://fallback.invalid/v1/chat/completions",
                "REASONING_EFFORT": "low",
                "REASONING_THINKING": "adaptive",
            },
            clear=True,
        ), patch.object(_API_CALLER.requests, "post", side_effect=responses) as post:
            result = _API_CALLER.call_vision_api(
                "MiniMax-M3.1-Flash",
                messages,
                sampling_params,
                max_retries=1,
                tools=tools,
            )

        self.assertEqual(result, "answer")
        self.assertEqual(post.call_count, 2)
        primary = post.call_args_list[0].kwargs["json"]
        fallback = post.call_args_list[1].kwargs["json"]
        self.assertEqual(primary["model"], "MiniMax-M3.1-Flash")
        self.assertEqual(primary["reasoning_effort"], "low")
        self.assertNotIn("thinking", primary)
        self.assertEqual(fallback["model"], "MiniMax-M3")
        self.assertEqual(fallback["thinking"], {"type": "adaptive"})
        self.assertNotIn("reasoning_effort", fallback)
        self.assertEqual(primary["messages"], messages)
        self.assertEqual(primary["tools"], tools)

    def test_preserves_reasoning_content_in_response_message(self):
        message = {"content": "answer", "reasoning_content": "reasoning"}
        response = _Response({"choices": [{"message": message}]})

        result, is_429, error_type = _parse_api_response(response, "MiniMax")

        self.assertEqual(result, message)
        self.assertFalse(is_429)
        self.assertIsNone(error_type)


if __name__ == "__main__":
    unittest.main()
