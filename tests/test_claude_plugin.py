import base64
import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from rplugin.python3.ai_plugin import AIPlugin


class TestAIPlugin(unittest.TestCase):
    def setUp(self):
        self.nvim = MagicMock()
        self.plugin = AIPlugin(self.nvim)

    @patch('rplugin.python3.ai_plugin.os.path.exists')
    @patch('rplugin.python3.ai_plugin.open', create=True)
    @patch('rplugin.python3.ai_plugin.json.load')
    def test__init__with_config_file(self, mock_json_load, mock_open, mock_path_exists):
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "current_model": "test-model",
            "filename_model": "test-filename-model",
            "max_tokens": 1000,
            "max_context_tokens": 2000,
            "truncate_conversation": False,
            "system_prompt": "Test prompt",
            "temperature": 0.5,
            "timeout": 30.0,
            "limit_window": 30.0,
        }

        self.plugin.__init__(self.nvim)

        self.assertEqual(self.plugin.current_model, "test-model")
        self.assertEqual(self.plugin.filename_model, "test-filename-model")
        self.assertEqual(self.plugin.max_tokens, 1000)
        self.assertEqual(self.plugin.max_context_tokens, 2000)
        self.assertFalse(self.plugin.truncate_conversation)
        self.assertEqual(self.plugin.system_prompt, "Test prompt")
        self.assertEqual(self.plugin.temperature, 0.5)
        self.assertEqual(self.plugin.timeout, 30.0)
        self.assertEqual(self.plugin.limit_window, 30.0)

    @patch('rplugin.python3.ai_plugin.os.path.exists')
    @patch('rplugin.python3.ai_plugin.open', create=True)
    @patch('rplugin.python3.ai_plugin.json.load')
    def test__init__without_config_file(self, mock_json_load, mock_open, mock_path_exists):
        mock_path_exists.return_value = False

        self.plugin.__init__(self.nvim)

        self.assertEqual(self.plugin.current_model, "anthropic/claude-3.5-haiku-20241022")
        self.assertEqual(self.plugin.filename_model, "anthropic/claude-3.5-haiku-20241022")
        self.assertEqual(self.plugin.max_tokens, 4096)
        self.assertEqual(self.plugin.max_context_tokens, 204800)
        self.assertTrue(self.plugin.truncate_conversation)
        self.assertIn("Please format all responses with line breaks at 80 columns.", self.plugin.system_prompt)
        self.assertEqual(self.plugin.temperature, 0.25)
        self.assertEqual(self.plugin.timeout, 60.0)
        self.assertEqual(self.plugin.limit_window, 60.0)

    def test__extract_code_blocks(self):
        text = "Here is some code:\n```python\nprint('Hello, world!')\n```\nAnd some more text."
        expected = [('python', "", "print('Hello, world!')\n", 1)]
        result = self.plugin._extract_code_blocks(text, 0)
        self.assertEqual(result, expected)

    def test__format_response(self):
        response_chunk = 'data: {"choices":[{"delta":{"content":"This is a response."}}]}'
        result = self.plugin._format_response(response_chunk)
        self.assertEqual(result, "This is a response.")

    @patch('rplugin.python3.ai_plugin.os.path.exists')
    @patch('rplugin.python3.ai_plugin.os.makedirs')
    @patch('rplugin.python3.ai_plugin.open', create=True)
    @patch('rplugin.python3.ai_plugin.json.dump')
    def test__save_attributes_to_file(self, mock_json_dump, mock_open, mock_makedirs, mock_path_exists):
        mock_path_exists.return_value = True
        mock_file = mock_open.return_value.__enter__.return_value

        self.plugin.current_model = "test-model"
        self.plugin.filename_model = "test-filename-model"
        self.plugin.max_tokens = 1000
        self.plugin.max_context_tokens = 2000
        self.plugin.truncate_conversation = False
        self.plugin.system_prompt = "Test prompt"
        self.plugin.temperature = 0.5
        self.plugin.timeout = 30.0
        self.plugin.limit_window = 30.0

        self.plugin._save_attributes_to_file()

        mock_makedirs.assert_called_once_with(os.path.join(
            os.path.expanduser('~'), '.config', 'nvim'), exist_ok=True)
        mock_open.assert_called_once_with(os.path.join(
            os.path.expanduser('~'), '.config', 'nvim', 'nvim_ai.json'), 'w')
        mock_json_dump.assert_called_once_with({
            "current_model": "test-model",
            "filename_model": "test-filename-model",
            "max_tokens": 1000,
            "max_context_tokens": 2000,
            "truncate_conversation": False,
            "system_prompt": "Test prompt",
            "temperature": 0.5,
            "timeout": 30.0,
            "limit_window": 30.0
        }, mock_file, indent=2)

    def test__get_filename_from_response(self):
        response = {
            "choices": [{
                "message": {
                    "content": "example_filename.txt"
                }
            }]
        }
        expected = "example_filename.txt"
        result = self.plugin._get_filename_from_response(response)
        self.assertEqual(result, expected)

    def test__count_tokens(self):
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine, thank you! How can I help you today?"}
        ]

        # Mock the http client responses
        mock_responses = [
            MagicMock(json=lambda: {"input_tokens": 10}),  # System prompt + first message
            MagicMock(json=lambda: {"input_tokens": 5}),   # First message
            MagicMock(json=lambda: {"input_tokens": 8})    # Second message
        ]
        self.plugin.http_client.post = MagicMock(side_effect=mock_responses)

        prompt_tokens, message_tokens = self.plugin._count_tokens(messages)

        self.assertEqual(prompt_tokens, 5)  # 10 - 5 (system + first message - first message)
        self.assertEqual(message_tokens, [5, 8])

    def test__count_tokens__empty_messages(self):
        messages = []
        prompt_tokens, message_tokens = self.plugin._count_tokens(messages)
        self.assertEqual(prompt_tokens, 0)
        self.assertEqual(message_tokens, [])

    def test__find_last_response__normal_case(self):
        buffer_content = (
            "<user>\nSome text before\n</user>\n"
            "<assistant>\nResponse 1\n</assistant>\n"
            "<user>\nSome text in between\n</user>\n"
            "<assistant>\nResponse 2\n</assistant>\n"
        )
        self.plugin.nvim.current.window.cursor = [10, 0]
        last_response, start_line = self.plugin._find_last_response(buffer_content)
        self.assertEqual(last_response, "Response 2")
        self.assertEqual(start_line, 9)

    def test__process_content__text_only(self):
        input_string = "This is a simple text without any content tags."
        expected_output = [
            {"type": "text", "text": "This is a simple text without any content tags."}
        ]
        result = self.plugin._process_content(input_string)
        self.assertEqual(result, expected_output)

    @patch('rplugin.python3.ai_plugin.open', new_callable=unittest.mock.mock_open, read_data=b"image data")
    @patch('rplugin.python3.ai_plugin.os.path.expanduser', side_effect=lambda x: x)
    @patch('rplugin.python3.ai_plugin.mimetypes.guess_type', return_value=('image/jpeg', None))
    def test__process_content__image(self, mock_guess_type, mock_expanduser, mock_open):
        input_string = "Here is an image: <content>/path/to/image.jpg</content>"
        expected_output = [
            {"type": "text", "text": "Here is an image: "},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(b"image data").decode('utf-8')
                }
            }
        ]
        result = self.plugin._process_content(input_string)
        self.assertEqual(result, expected_output)

    def test__truncate_conversation(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        # Mock _count_tokens to return values that would trigger truncation
        self.plugin._count_tokens = MagicMock(return_value=(10, [5, 15, 20]))  # 60 total tokens

        result = self.plugin._truncate_conversation(messages, 40)  # Max 40 tokens
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], messages[-1])

    def test__make_completion_request(self):
        messages = [{"role": "user", "content": "Hello"}]
        mock_response = MagicMock()
        self.plugin.http_client.request = MagicMock(return_value=mock_response)

        response = self.plugin._make_completion_request(
            messages=messages,
            stream=False,
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            system="Test system prompt"
        )

        self.plugin.http_client.request.assert_called_once_with(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "test-model",
                "messages": [{"role": "system", "content": "Test system prompt"}] + messages,
                "stream": False,
                "max_tokens": 100,
                "temperature": 0.5,
            },
            timeout=60.0
        )
        self.assertEqual(response, mock_response)

    def test__make_completion_request__streaming(self):
        messages = [{"role": "user", "content": "Hello"}]
        mock_response = MagicMock()
        self.plugin.http_client.stream = MagicMock(return_value=mock_response)

        response = self.plugin._make_completion_request(
            messages=messages,
            stream=True
        )

        self.plugin.http_client.stream.assert_called_once()
        self.assertEqual(response, mock_response)

    def test_get_available_models(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model1"},
                {"id": "model2"},
                {"id": "anthropic/claude-3.5-sonnet-20240620"}
            ]
        }
        self.plugin.http_client.get = MagicMock(return_value=mock_response)

        models = self.plugin.get_available_models()
        
        expected_models = [
            "model1",
            "model2",
            "anthropic/claude-3.5-sonnet-20240620",
            "anthropic/claude-3.5-sonnet-20241022"
        ]
        self.assertEqual(sorted(models), sorted(expected_models))


if __name__ == '__main__':
    unittest.main()
