import base64
import os
import unittest
from unittest.mock import MagicMock, patch

import anthropic

from rplugin.python3.claude_plugin import ClaudePlugin


class TestClaudePlugin(unittest.TestCase):
    def setUp(self):
        self.nvim = MagicMock()
        self.plugin = ClaudePlugin(self.nvim)

    @patch('rplugin.python3.claude_plugin.os.path.exists')
    @patch('rplugin.python3.claude_plugin.open', create=True)
    @patch('rplugin.python3.claude_plugin.json.load')
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
        }

        self.plugin.__init__(self.nvim)

        self.assertEqual(self.plugin.current_model, "test-model")
        self.assertEqual(self.plugin.filename_model, "test-filename-model")
        self.assertEqual(self.plugin.max_tokens, 1000)
        self.assertEqual(self.plugin.max_context_tokens, 2000)
        self.assertFalse(self.plugin.truncate_conversation)
        self.assertEqual(self.plugin.system_prompt, "Test prompt")
        self.assertEqual(self.plugin.temperature, 0.5)

    @patch('rplugin.python3.claude_plugin.os.path.exists')
    @patch('rplugin.python3.claude_plugin.open', create=True)
    @patch('rplugin.python3.claude_plugin.json.load')
    def test__init__without_config_file(self, mock_json_load, mock_open, mock_path_exists):
        mock_path_exists.return_value = False

        self.plugin.__init__(self.nvim)

        self.assertEqual(self.plugin.current_model,
                         "claude-3-5-haiku-20241022")
        self.assertEqual(self.plugin.max_tokens, 4096)
        self.assertEqual(self.plugin.max_context_tokens, 204800)
        self.assertTrue(self.plugin.truncate_conversation)
        self.assertIn(
            "Please format all responses with line breaks at 80 columns.", self.plugin.system_prompt)
        self.assertEqual(self.plugin.temperature, 0.25)

    def test__extract_code_blocks(self):
        text = "Here is some code:\n```python\nprint('Hello, world!')\n```\nAnd some more text."
        expected = [('python', "", "print('Hello, world!')\n", 1)]
        result = self.plugin._extract_code_blocks(text, 0)
        self.assertEqual(result, expected)

    def test__format_response(self):
        response = [
            anthropic.types.TextBlock(type="text", text="This is a response."),
            anthropic.types.TextBlock(
                type="text", text="This is another response.")
        ]
        result = self.plugin._format_response(response)
        self.assertIn("This is a response.", result)
        self.assertIn("This is another response.", result)

    @patch('rplugin.python3.claude_plugin.os.path.exists')
    @patch('rplugin.python3.claude_plugin.os.makedirs')
    @patch('rplugin.python3.claude_plugin.open', create=True)
    @patch('rplugin.python3.claude_plugin.json.dump')
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

        self.plugin._save_attributes_to_file()

        mock_makedirs.assert_called_once_with(os.path.join(
            os.path.expanduser('~'), '.config', 'nvim'), exist_ok=True)
        mock_open.assert_called_once_with(os.path.join(
            os.path.expanduser('~'), '.config', 'nvim', 'nvim_claude.json'), 'w')
        mock_json_dump.assert_called_once_with({
            "current_model": "test-model",
            "filename_model": "test-filename-model",
            "max_tokens": 1000,
            "max_context_tokens": 2000,
            "truncate_conversation": False,
            "system_prompt": "Test prompt",
            "temperature": 0.5,
            "timeout": 300.0,
            "limit_window": 60.0
        }, mock_file, indent=2)

    def test__extract_code_blocks__multiple_blocks(self):
        text = """
        Here is some code:
        ```python
        print('Hello, world!')
        ```
        And here is some more code:
        ```javascript
        console.log('Hello, world!');
        ```
        """
        expected = [
            ('python', "", "        print('Hello, world!')\n        ", 2),
            ('javascript', "", "        console.log('Hello, world!');\n        ", 6)
        ]
        result = self.plugin._extract_code_blocks(text, 0)
        self.assertEqual(result, expected)

    def test__extract_code_blocks__no_language_specified(self):
        text = """
        Here is some code:
        ```
        print('Hello, world!')
        ```
        """
        expected = [
            ('txt', '', "        print('Hello, world!')\n        ", 2),
        ]
        result = self.plugin._extract_code_blocks(text, 0)
        self.assertEqual(result, expected)

    def test__extract_code_blocks__no_code_blocks(self):
        text = "Here is some text without any code blocks."
        expected = []
        result = self.plugin._extract_code_blocks(text, 0)
        self.assertEqual(result, expected)

    def test__extract_code_blocks__empty_code_block(self):
        text = """
        Here is an empty code block:
        ```python
        ```
        """
        expected = [('python', "", "        ", 2)]
        result = self.plugin._extract_code_blocks(text, 0)
        self.assertEqual(result, expected)

    def test__format_response__text_blocks(self):
        response = [
            anthropic.types.TextBlock(type="text", text="Hello, world!"),
            anthropic.types.TextBlock(type="text", text="How are you?")
        ]
        expected = "Hello, world!How are you?"
        result = self.plugin._format_response(response)
        self.assertEqual(result, expected)

    def test__format_response__empty_response(self):
        response = []
        expected = ""
        result = self.plugin._format_response(response)
        self.assertEqual(result, expected)

    def test__format_response__non_text_blocks(self):
        response = [
            anthropic.types.TextBlock(type="text", text="Hello, world!"),
            # Assuming ToolUseBlock is another type of block
            anthropic.types.ToolUseBlock(
                id="example_id",
                input="example_input",
                name="example_tool",
                type="tool_use"
            )
        ]
        expected = "Hello, world!"
        result = self.plugin._format_response(response)
        self.assertEqual(result, expected)

    def test__get_filename_from_response(self):
        response = anthropic.types.Message(
            id="example_id",
            content=[
                anthropic.types.TextBlock(
                    type="text", text="example_filename.txt")
            ],
            model="claude-3-5-sonnet-20240620",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic.types.Usage(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12
            )
        )
        expected = "example_filename.txt"
        result = self.plugin._get_filename_from_response(response)
        self.assertEqual(result, expected)

    def test__get_filename_from_response__spaces(self):
        response = anthropic.types.Message(
            id="example_id",
            content=[
                anthropic.types.TextBlock(
                    type="text", text="example filename with spaces.txt")
            ],
            model="claude-3-5-sonnet-20240620",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic.types.Usage(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12
            )
        )
        expected = "'example filename with spaces.txt'"
        result = self.plugin._get_filename_from_response(response)
        self.assertEqual(result, expected)

    def test__get_filename_from_response__special_characters(self):
        response = anthropic.types.Message(
            id="example_id",
            content=[
                anthropic.types.TextBlock(
                    type="text", text="example@filename#with$special&characters.txt")
            ],
            model="claude-3-5-sonnet-20240620",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic.types.Usage(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12
            )
        )
        expected = "'example@filename#with$special&characters.txt'"
        result = self.plugin._get_filename_from_response(response)
        self.assertEqual(result, expected)

    def test__get_filename_from_response__newlines(self):
        response = anthropic.types.Message(
            id="example_id",
            content=[
                anthropic.types.TextBlock(
                    type="text", text="example_filename_with\nnewlines.txt")
            ],
            model="claude-3-5-sonnet-20240620",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic.types.Usage(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12
            )
        )
        expected = "example_filename_withnewlines.txt"
        result = self.plugin._get_filename_from_response(response)
        self.assertEqual(result, expected)

    def test__get_claude_models(self):
        models = self.plugin.get_claude_models()
        self.assertIsInstance(iter(models), type(iter([])))
        self.assertTrue(all(isinstance(model, str) for model in models))

    def test__count_tokens(self):
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant",
                "content": "I'm fine, thank you! How can I help you today?"}
        ]
        self.plugin.current_model = "claude-3-5-sonnet-20240620"
        self.plugin.system_prompt = "This is a system prompt."

        # Mock the client response for token counting
        self.plugin.claude_client.beta.messages.count_tokens = unittest.mock.Mock(side_effect=[
            unittest.mock.Mock(input_tokens=10),
            unittest.mock.Mock(input_tokens=5),
            unittest.mock.Mock(input_tokens=8)
        ])

        prompt_tokens, message_tokens = self.plugin._count_tokens(messages)

        # 10 (system + first message) - 5 (first message)
        self.assertEqual(prompt_tokens, 5)
        self.assertEqual(message_tokens[0].input_tokens, 5)
        self.assertEqual(message_tokens[1].input_tokens, 8)

    def test__count_tokens__empty_messages(self):
        messages = []
        self.plugin.current_model = "claude-3-5-sonnet-20240620"
        self.plugin.system_prompt = "This is a system prompt."

        # Mock the client response for token counting
        self.plugin.claude_client.beta.messages.count_tokens = unittest.mock.Mock(
            return_value=unittest.mock.Mock(input_tokens=0))

        prompt_tokens, message_tokens = self.plugin._count_tokens(messages)

        self.assertEqual(prompt_tokens, 0)
        self.assertEqual(message_tokens, [])

    def test__count_tokens__single_message(self):
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        self.plugin.current_model = "claude-3-5-sonnet-20240620"
        self.plugin.system_prompt = "This is a system prompt."

        # Mock the client response for token counting
        self.plugin.claude_client.beta.messages.count_tokens = unittest.mock.Mock(side_effect=[
            unittest.mock.Mock(input_tokens=10),
            unittest.mock.Mock(input_tokens=5)
        ])

        prompt_tokens, message_tokens = self.plugin._count_tokens(messages)

        # 10 (system + first message) - 5 (first message)
        self.assertEqual(prompt_tokens, 5)
        self.assertEqual(message_tokens[0].input_tokens, 5)

    def test__find_last_response__normal_case(self):
        buffer_content = (
            "<user>\n"
            "Some text before\n"
            "</user>\n"
            "<assistant>\n"
            "Response 1 line 1\n"
            "Response 1 line 2\n"
            "</assistant>\n"
            "<user>\n"
            "Some text in between\n"
            "</user>\n"
            "<assistant>\n"
            "Response 2 line 1\n"
            "Response 2 line 2\n"
            "</assistant>\n"
        )
        self.plugin.nvim.current.window.cursor = [14, 0]
        last_response, start_line = self.plugin._find_last_response(
            buffer_content)
        self.assertEqual(last_response, "Response 2 line 1\nResponse 2 line 2")
        self.assertEqual(start_line, 10)

    def test__find_last_response__normal_case__cursor_above(self):
        buffer_content = (
            "<user>\n"
            "Some text before\n"
            "</user>\n"
            "<assistant>\n"
            "Response 1 line 1\n"
            "Response 1 line 2\n"
            "</assistant>\n"
            "<user>\n"
            "Some text in between\n"
            "</user>\n"
            "<assistant>\n"
            "Response 2 line 1\n"
            "Response 2 line 2\n"
            "</assistant>\n"
        )
        self.plugin.nvim.current.window.cursor = [5, 0]
        last_response, start_line = self.plugin._find_last_response(
            buffer_content)
        self.assertEqual(last_response, "Response 1 line 1\nResponse 1 line 2")
        self.assertEqual(start_line, 3)

    def test__find_last_response__with_assistant_tag(self):
        buffer_content = (
            "Some text before\n"
            "<assistant>Response 1</assistant>\n"
            "Some text in between\n"
            "<assistant>Response 2</assistant>\n"
            "Some text after\n"
        )
        # Cursor on the line with "Some text in between"
        self.plugin.nvim.current.window.cursor = [4, 0]
        last_response, start_line = self.plugin._find_last_response(
            buffer_content)
        self.assertEqual(last_response, "Response 2")
        self.assertEqual(start_line, 3)

    def test__find_last_response__within_assistant_block(self):
        buffer_content = (
            "Some text before\n"
            "<assistant>Response 1\n"
            "Continued response</assistant>\n"
            "Some text after\n"
        )
        self.plugin.nvim.current.window.cursor = [
            2, 0]  # Cursor within the assistant block
        last_response, start_line = self.plugin._find_last_response(
            buffer_content)
        self.assertEqual(last_response, "Response 1\nContinued response")
        self.assertEqual(start_line, 1)

    def test__find_last_response__no_assistant_tag(self):
        buffer_content = (
            "Some text before\n"
            "Some text in between\n"
            "Some text after\n"
        )
        # Cursor on the line with "Some text in between"
        self.plugin.nvim.current.window.cursor = [2, 0]
        last_response, start_line = self.plugin._find_last_response(
            buffer_content)
        self.assertIsNone(last_response)
        self.assertIsNone(start_line)

    def test__find_last_response__assistant_tag_on_same_line(self):
        buffer_content = (
            "Some text before\n"
            "<assistant>Response 1</assistant> Some text after\n"
        )
        # Cursor on the line with the assistant tag
        self.plugin.nvim.current.window.cursor = [2, 0]
        last_response, start_line = self.plugin._find_last_response(
            buffer_content)
        self.assertEqual(last_response, "Response 1")
        self.assertEqual(start_line, 1)

    def test__generate_filename__python(self):
        language = "python"
        code = "def hello_world():\n    print('Hello, world!')"
        expected_filename = "hello_world.py"

        self.plugin.claude_client.messages.create = lambda *args, **kwargs: MagicMock(
            content=[
                anthropic.types.TextBlock(type="text", text=expected_filename)
            ]
        )

        filename = self.plugin._generate_filename(language, None, code)
        self.assertEqual(filename, expected_filename)

    def test__generate_filename__javascript(self):
        language = "javascript"
        code = "function helloWorld() {\n    console.log('Hello, world!');\n}"
        expected_filename = "helloWorld.js"

        self.plugin.claude_client.messages.create = lambda *args, **kwargs: MagicMock(
            content=[
                anthropic.types.TextBlock(type="text", text=expected_filename)
            ]
        )

        filename = self.plugin._generate_filename(language, None, code)
        self.assertEqual(filename, expected_filename)

    def test__generate_filename__no_language(self):
        language = ""
        code = "print('Hello, world!')"
        expected_filename = "hello_world.txt"

        self.plugin.claude_client.messages.create = lambda *args, **kwargs: MagicMock(
            content=[
                anthropic.types.TextBlock(type="text", text=expected_filename)
            ]
        )

        filename = self.plugin._generate_filename(language, None, code)
        self.assertEqual(filename, expected_filename)

    def test__generate_filename__special_characters(self):
        language = "python"
        code = "def special_chars():\n    print('Hello, @world!')"
        expected_filename = "special_chars.py"

        self.plugin.claude_client.messages.create = lambda *args, **kwargs: MagicMock(
            content=[
                anthropic.types.TextBlock(type="text", text=expected_filename)
            ]
        )

        filename = self.plugin._generate_filename(language, None, code)
        self.assertEqual(filename, expected_filename)

    def test__generate_filename__with_existing_filename(self):
        language = "python"
        filename = "existing_file.py"
        code = "print('Hello, world!')"

        # Create a mock for the client
        self.plugin.claude_client = MagicMock()
        self.plugin.claude_client.messages.create = MagicMock()

        # Should return the existing filename without calling Claude
        result = self.plugin._generate_filename(language, filename, code)
        self.assertEqual(result, filename)
        self.plugin.claude_client.messages.create.assert_not_called()

    def test__generate_filename__with_empty_filename(self):
        language = "python"
        filename = ""
        code = "print('Hello, world!')"
        expected_filename = "hello.py"

        self.plugin.claude_client.messages.create = lambda *args, **kwargs: MagicMock(
            content=[
                anthropic.types.TextBlock(type="text", text=expected_filename)
            ]
        )

        result = self.plugin._generate_filename(language, filename, code)
        self.assertEqual(result, expected_filename)

    def test__generate_filename__with_none_filename(self):
        language = "python"
        filename = None
        code = "print('Hello, world!')"
        expected_filename = "hello.py"

        self.plugin.claude_client.messages.create = lambda *args, **kwargs: MagicMock(
            content=[
                anthropic.types.TextBlock(type="text", text=expected_filename)
            ]
        )

        result = self.plugin._generate_filename(language, filename, code)
        self.assertEqual(result, expected_filename)

    def test__process_content___text_only(self):
        input_string = "This is a simple text without any content tags."
        expected_output = [
            {"type": "text", "text": "This is a simple text without any content tags."}]

        result = self.plugin._process_content(input_string)
        self.assertEqual(result, expected_output)

    @patch('rplugin.python3.claude_plugin.open', new_callable=unittest.mock.mock_open, read_data=b"image data")
    @patch('rplugin.python3.claude_plugin.os.path.expanduser', side_effect=lambda x: x)
    @patch('rplugin.python3.claude_plugin.mimetypes.guess_type', return_value=('image/jpeg', None))
    def test__process_content___image(self, mock_guess_type, mock_expanduser, mock_open):
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

    @patch('rplugin.python3.claude_plugin.open', new_callable=unittest.mock.mock_open, read_data=b"file data")
    @patch('rplugin.python3.claude_plugin.os.path.expanduser', side_effect=lambda x: x)
    @patch('rplugin.python3.claude_plugin.mimetypes.guess_type', return_value=('image/jpeg', None))
    def test__process_content__unknown_file_type(self, mock_guess_type, mock_expanduser, mock_open):
        input_string = "Here is a file: <content>/path/to/file.unknown</content>"
        expected_output = [
            {"type": "text", "text": "Here is a file: "},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(b"file data").decode('utf-8')
                }
            }
        ]

        result = self.plugin._process_content(input_string)
        self.assertEqual(result, expected_output)

    @patch('rplugin.python3.claude_plugin.os.path.expanduser', side_effect=lambda x: x)
    def test__process_content__nonexistent_file(self, mock_expanduser):
        input_string = "Here is a nonexistent file: <content>/path/to/nonexistent.file</content>"
        expected_output = [
            {"type": "text", "text": "Here is a nonexistent file: "},
            {"type": "text", "text": "<content>/path/to/nonexistent.file</content>"}
        ]

        result = self.plugin._process_content(input_string)
        self.assertEqual(result, expected_output)

    def test__truncate_conversation_no_messages(self):
        messages = []
        result = self.plugin._truncate_conversation(messages, 100)
        self.assertEqual(result, messages)

    def test__truncate_conversation_single_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = self.plugin._truncate_conversation(messages, 100)
        self.assertEqual(result, messages)

    @patch.object(ClaudePlugin, '_count_tokens', return_value=(
        10, [
            MagicMock(input_tokens=5),
            MagicMock(input_tokens=15),
            MagicMock(input_tokens=20)
        ]))
    def test__truncate_conversation_within_limit(self, mock_count_tokens):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = self.plugin._truncate_conversation(messages, 50)
        self.assertEqual(result, messages)

    @patch.object(ClaudePlugin, '_count_tokens', return_value=(
        10, [
            MagicMock(input_tokens=5),
            MagicMock(input_tokens=15),
            MagicMock(input_tokens=20)
        ]))
    def test__truncate_conversation_exceeds_limit(self, mock_count_tokens):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        expected_result = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = self.plugin._truncate_conversation(messages, 45)
        self.assertEqual(result, expected_result)

    @patch.object(ClaudePlugin, '_count_tokens', return_value=(
        10, [
            MagicMock(input_tokens=5),
            MagicMock(input_tokens=15),
            MagicMock(input_tokens=20)
        ]))
    def test__truncate_conversation_exceeds_limit_truncate_more(self, mock_count_tokens):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        expected_result = [
            {"role": "user", "content": "How are you?"}
        ]
        result = self.plugin._truncate_conversation(messages, 20)
        self.assertEqual(result, expected_result)

    def test__wrap_new_content_with_user_tags_no_previous_response(self):
        self.plugin.nvim.current.buffer = MagicMock()
        self.plugin.nvim.current.buffer.__getitem__.return_value = [
            "This is a new message without previous response."
        ]
        self.plugin._wrap_new_content_with_user_tags()
        expected_buffer = [
            "<user>",
            "This is a new message without previous response.",
            "</user>"
        ]
        self.plugin.nvim.current.buffer.__setitem__.assert_called_once_with(
            slice(None, None, None), expected_buffer
        )

    def test__wrap_new_content_with_user_tags_with_previous_response(self):
        self.plugin.nvim.current.buffer = MagicMock()
        self.plugin.nvim.current.buffer.__getitem__.return_value = [
            "<assistant>",
            "This is a response from the assistant.",
            "</assistant>",
            "This is a new message."
        ]
        self.plugin._wrap_new_content_with_user_tags()
        expected_buffer = [
            "<assistant>",
            "This is a response from the assistant.",
            "</assistant>",
            "",
            "<user>",
            "This is a new message.",
            "</user>"
        ]
        self.plugin.nvim.current.buffer.__setitem__.assert_called_once_with(
            slice(None, None, None), expected_buffer
        )

    def test__wrap_new_content_with_user_tags_already_wrapped(self):
        self.plugin.nvim.current.buffer = MagicMock()
        self.plugin.nvim.current.buffer.__getitem__.return_value = [
            "<user>",
            "This is already wrapped content.",
            "</user>"
        ]
        result = self.plugin._wrap_new_content_with_user_tags()
        expected_buffer = [
            "<user>",
            "This is already wrapped content.",
            "</user>"
        ]
        self.assertIsNone(result)
        self.plugin.nvim.current.buffer.__setitem__.assert_not_called()

    def test_replace_prompt_command_with_current_buffer(self):
        self.plugin.nvim.current.buffer = MagicMock()
        self.plugin.nvim.current.buffer.__getitem__.return_value = [
            "New system prompt content."
        ]
        self.plugin.nvim.current.buffer.number = 1

        mock_path = 'rplugin.python3.claude_plugin.ClaudePlugin._save_attributes_to_file'
        with patch(mock_path) as mock_save_attributes:
            self.plugin.replace_prompt_command([])

        self.assertEqual(self.plugin.system_prompt,
                         "New system prompt content.")
        self.plugin.nvim.out_write.assert_called_once_with(
            "System prompt replaced with contents of buffer 1.\n"
        )
        mock_save_attributes.assert_called_once()

    def test_replace_prompt_command_with_specified_buffer(self):
        buffer_mock = MagicMock()
        buffer_mock.__getitem__.return_value = [
            "New system prompt content from specified buffer."
        ]
        buffer_mock.number = 2

        class BufferDict(dict):
            def __iter__(self):
                return iter(self.values())

        self.plugin.nvim.buffers = BufferDict({2: buffer_mock})

        mock_path = 'rplugin.python3.claude_plugin.ClaudePlugin._save_attributes_to_file'
        with patch(mock_path) as mock_save_attributes:
            self.plugin.replace_prompt_command([2])

        self.assertEqual(self.plugin.system_prompt,
                         "New system prompt content from specified buffer.")
        self.plugin.nvim.out_write.assert_called_once_with(
            "System prompt replaced with contents of buffer 2.\n"
        )
        mock_save_attributes.assert_called_once()

    def test_replace_prompt_command_with_nonexistent_buffer(self):
        class BufferDict(dict):
            def __iter__(self):
                return iter(self.values())

        self.plugin.nvim.buffers = BufferDict({1: MagicMock()})

        mock_path = 'rplugin.python3.claude_plugin.ClaudePlugin._save_attributes_to_file'
        with patch(mock_path) as mock_save_attributes:
            self.plugin.replace_prompt_command([99])

        self.plugin.nvim.err_write.assert_called_once_with(
            "Error: Buffer 99 does not exist.\n"
        )
        mock_save_attributes.assert_not_called()

    def test_replace_prompt_command_with_exception(self):
        self.plugin.nvim.current.buffer = MagicMock()
        self.plugin.nvim.current.buffer.__getitem__.side_effect = Exception(
            "Test exception")

        mock_path = 'rplugin.python3.claude_plugin.ClaudePlugin._save_attributes_to_file'
        with patch(mock_path) as mock_save_attributes:
            self.plugin.replace_prompt_command([])

        self.assertEqual(self.plugin.nvim.err_write.call_count, 1)
        self.assertIn(
            "Exception: Test exception",
            self.plugin.nvim.err_write.call_args[0][0].splitlines()[-2]
        )
        mock_save_attributes.assert_not_called()

    def test_replace_prompt_command_with_file(self):
        filename = "test_file.txt"
        with open(filename, 'w') as f:
            f.write("New system prompt content from file.")
        self.plugin.replace_prompt_command([filename])
        self.assertEqual(self.plugin.system_prompt,
                         "New system prompt content from file.")
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
