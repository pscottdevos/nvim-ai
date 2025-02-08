"""
Claude Neovim Plugin - AI integration for Neovim
Copyright (C) [2024] [P. Scott DeVos]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import json
import base64
import mimetypes
import os
import re
import shlex
from datetime import datetime
from traceback import format_exc
from typing import Callable, Optional, List, Tuple

import anthropic
import pynvim
from httpx import Timeout

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')


# Pattern to match :b<buffer_number> with at least one surrounding whitespace
BUFFER_PATTERN = re.compile(r'\s+:b(\d+)\s+')
# Pattern to match material between <content> tags
CONTENT_PATTERN = re.compile(r'<content>(.*?)</content>', re.DOTALL)

# Maximum number of tokens Claude will return
DEFAULT_MAX_TOKENS = 4096
ABSOLUTE_MAX_TOKENS = 8192
ABSOLUTE_MIN_TOKENS = 128

DEFAULT_TEMPERATURE = 0.25
DEFAULT_TIMEOUT = 60.0

# Maximum number of tokens to send to Claude
ABSOLUTE_MIN_CONTEXT_TOKENS = 1024
ABSOLUTE_MAX_CONTEXT_TOKENS = 1024 * 200
DEFAULT_MAX_CONTEXT_TOKENS = ABSOLUTE_MAX_CONTEXT_TOKENS

SETTINGS_FILE = 'nvim_claude.json'

DEFAULT_SYSTEM_PROMPT = """
    You are a highly skilled software engineer. Your task when asked to generate
    code is to generate code that is:
    - Clear and Readable: Use descriptive variable and function names, and
      include comments where necessary to explain complex logic.
    - Concise: Avoid unnecessary code and keep the implementation as simple
      as possible.
    - Well-Designed: Follow best practices for the language, including proper
      use of data structures, error handling, and modular design.
    - Efficient: Optimize for performance without sacrificing readability.
    - Consistent: Adhere to the coding style and conventions of the language.

    The following applies only to code in your responses:
    - While code should generally break at 80 columns, avoid breaking
        lines if it would make the code awkward or less readable.
    - Use standard language-specific indentation:
        * 4 spaces for Python
        * 2 spaces for JavaScript, Typescript, CSS, HTML, XML, and JSON
        * Follow established conventions for other languages
    - Separate code blocks from other text and from each other with
        blank lines
    - Code Quality Principles:
        * Prioritize readability and clarity
        * Use meaningful, descriptive variable and function names
        * Write self-documenting code
        * Include concise, informative comments when they add value
    - Demonstrate language-specific best practices
        - When providing multiple code examples:
            * Show different approaches
            * Highlight alternative implementations
            * Explain trade-offs between approaches

    You are also a highly-intelligent, helpful assistant. If you are asked
    to do something that is not code generation, you will do it.

    Please format all responses with line breaks at 80 columns. Not just code
    lines, but all lines. This is important!
"""


@pynvim.plugin
class ClaudePlugin:
    """A Neovim plugin that provides integration with Anthropic's Claude AI
    model.

    This plugin enables users to interact with Claude directly from Neovim,
    offering features like code generation, text completion, and intelligent
    responses. It supports multiple Claude models, configurable settings, and
    maintains conversation context within buffers.

    The plugin provides commands for:
    - Text/code completion
    - Model selection and management
    - Code block extraction and file handling
    - Conversation management with context truncation
    """

    COMMENT_STYLES = {
        'python': '#',
        'javascript': '//',
        'js': '//',
        'c': '//',
        'cpp': '//',
        'asm': ';',
        'nasm': ';',
        'rust': '//',
        'go': '//',
    }

    def __init__(self, nvim):
        self.nvim = nvim
        self.buffer_filenames = {}

        # Default settings
        default_config = {
            "current_model": "claude-3-5-haiku-20241022",
            "filename_model": "claude-3-5-haiku-20241022",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_context_tokens": DEFAULT_MAX_CONTEXT_TOKENS,
            "truncate_conversation": True,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "temperature": DEFAULT_TEMPERATURE,
            "timeout": DEFAULT_TIMEOUT,
        }

        # Load configuration from file
        config_dir = os.path.join(os.path.expanduser('~'), '.config', 'nvim')
        attributes_path = os.path.join(config_dir, SETTINGS_FILE)

        # Initialize configuration with default settings
        config = default_config.copy()

        if os.path.exists(attributes_path):
            try:
                with open(attributes_path, 'r') as file:
                    file_config = json.load(file)
                    # Update default configuration with file configuration
                    config.update(file_config)
            except Exception as e:
                self.nvim.err_write(
                    f"Error loading configuration file:\n{format_exc()}\n")

        # Set attributes from the updated configuration
        self.current_model = config["current_model"]
        self.filename_model = config["filename_model"]
        self.max_tokens = config["max_tokens"]
        self.max_context_tokens = config["max_context_tokens"]
        self.truncate_conversation = config["truncate_conversation"]
        self.system_prompt = config["system_prompt"]
        self.temperature = config["temperature"]
        self.timeout = config["timeout"]
        self.claude_client = anthropic.Client(
            timeout=Timeout(10.0, read=self.timeout, write=self.timeout))

    def _save_attributes_to_file(self):
        attributes = {
            "current_model": self.current_model,
            "filename_model": self.filename_model,
            "max_tokens": self.max_tokens,
            "max_context_tokens": self.max_context_tokens,
            "truncate_conversation": self.truncate_conversation,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        config_dir = os.path.join(os.path.expanduser('~'), '.config', 'nvim')
        os.makedirs(config_dir, exist_ok=True)
        attributes_path = os.path.join(config_dir, SETTINGS_FILE)
        with open(attributes_path, 'w') as file:
            json.dump(attributes, file, indent=2)

    @classmethod
    def _extract_code_blocks(cls, text: str, start_line: int) -> List[Tuple[str, str, str, int]]:
        """Extracts code blocks from the given text, including filenames if present.

        Returns a list of tuples containing:
        - file_type: The language/type of the code block
        - filename: Optional filename if specified in a comment
        - code: The code content
        - line_number: The line number where the code block starts

        Handles comment styles for:
        - Python (#)
        - JavaScript (//)
        - C (//)
        - C++ (//)
        - Assembly (;)
        - Rust (//)
        - Go (//)
        """

        code_blocks = re.finditer(
            r'```(\w*)\n'  # Language identifier (note: * instead of +)
            r'(?:#!.*\n)?'  # Optional shebang
            # Comment with filename (note added \n)
            r'(?:[^`\n]*?(?:(?://|#|;)\s*([^\n]+)\n))?'
            r'([\s\S]*?)```',  # Rest of code block
            text)

        result = []
        for block in code_blocks:
            # Calculate the line number by counting newlines before the match
            line_number = start_line + text[:block.start()].count('\n')

            file_type = block.group(1) or 'txt'
            filename = block.group(2) or ''  # Filename from comment if present
            code = block.group(3)  # Preserve all whitespace

            # If no filename was found in the first line comment, look for it in the second line
            # in case there was a shebang
            if not filename and code:
                first_line = code.split('\n', 1)[0]
                comment_style = cls.COMMENT_STYLES.get(file_type.lower())
                if comment_style:
                    filename_match = re.match(
                        rf'^[^`\n]*?{comment_style}\s*([^\n]+)',
                        first_line
                    )
                    if filename_match:
                        filename = filename_match.group(1)
                        # Remove the filename line from the code if it was found
                        # code = code.split('\n', 1)[1] if '\n' in code else ''

            result.append((file_type, filename, code, line_number))
        return result

    @staticmethod
    def _format_response(response: list[
            anthropic.types.TextBlock | anthropic.types.ToolUseBlock
    ]) -> str:
        """Format the response with proper wrapping and tags."""
        # Extract text from response blocks
        text = ""
        for block in response:
            if isinstance(block, anthropic.types.TextBlock):
                text += block.text
            # Skip other block types for now
        return text

    @staticmethod
    def _get_filename_from_response(response: anthropic.types.Message) -> str:
        """Get a filename from the response."""
        filename = (
            ClaudePlugin._format_response(response.content)
            .replace('\n', '')
            .strip()
        )
        # Escape spaces and special characters in filename
        return shlex.quote(filename)

    def _count_tokens(self, messages: List[dict]) -> Tuple[int, int]:
        if not messages:
            return 0, []
        # Tokenize the system prompt and first message because we have to
        # include a message or anthropic will throw an error
        system_prompt_tokens_and_first_message = \
            self.claude_client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=self.current_model,
                system=self.system_prompt,
                messages=[messages[0]],
            )
        # Tokenize each message and store the token counts
        message_tokens = [
            self.claude_client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=self.current_model,
                system="",
                messages=[msg],
            )
            for msg in messages
        ]
        # Subtract the tokens of the first message because we included it
        # with the system prompt
        prompt_tokens = system_prompt_tokens_and_first_message.input_tokens - \
            message_tokens[0].input_tokens

        return prompt_tokens, message_tokens

    def _find_last_response(self, buffer_content: str) -> Tuple[Optional[str], Optional[int]]:
        cursor_pos = self.nvim.current.window.cursor[0] - 1
        buffer_lines = buffer_content.split('\n')

        # Check if the cursor is within or on an <assistant> tag
        for i in range(cursor_pos, -1, -1):
            line = buffer_lines[i]
            if '<assistant>' in line and '</assistant>' in line:
                start = line.find('<assistant>') + len('<assistant>')
                end = line.find('</assistant>')
                return line[start:end].strip(), i
            elif '<assistant>' in line:
                # Cursor is on the opening tag or within the block
                start = i
                for j in range(i, len(buffer_lines)):
                    if '</assistant>' in buffer_lines[j]:
                        end = j
                        return (
                            '\n'.join(buffer_lines[start:end + 1])
                            .replace('<assistant>', '')
                            .replace('</assistant>', '')
                            .strip(),
                            start
                        )

        # If no match is found, write an error
        self.nvim.err_write(
            "No <assistant> tag found around or above the cursor\n")
        return None, None

    def _generate_filename(self, language: str, filename: str, code: str) -> str:
        """Generate a filename for a code block."""
        # If filename is provided and not empty, use it
        if filename:
            return filename

        # Ask Claude to generate a filename
        messages = [
            {
                "role": "user",
                "content": (
                    f"Given this {language} code:\n\n{code}\n\n"
                    "Respond with only a filename that would be appropriate for this code. "
                    "The filename should be descriptive and follow standard conventions "
                    f"for {language} files. Do not include any explanation, just the filename."
                )
            }
        ]

        try:
            response = self.claude_client.messages.create(
                model=self.filename_model,
                max_tokens=128,
                temperature=0.0,
                system=self.system_prompt,
                messages=messages
            )
            return self._get_filename_from_response(response)
        except Exception as e:
            self.nvim.err_write(
                f"Error generating filename:\n{format_exc()}\n"
            )
            return f"untitled.{language}"

    def _process_content(self, input_string: str) -> List[dict]:
        """                                                                                                         
        Process input string, converting <content> file references to base64                                          
        encoded content or preserving text.                                                                         
        """
        result = []
        last_end = 0

        for match in CONTENT_PATTERN.finditer(input_string):
            # Add any text before the tag
            if match.start() > last_end:
                result.append({
                    "type": "text",
                    "text": input_string[last_end:match.start()]
                })

            # Process file reference
            filepath = match.group(1).strip()
            filepath = os.path.expanduser(filepath)

            try:
                media_type, _ = mimetypes.guess_type(filepath)
                if media_type and media_type.startswith('image'):
                    with open(filepath, 'rb') as file:
                        content = file.read()
                        result.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(content).decode('utf-8')
                            }
                        })
                elif media_type and media_type.startswith('video'):
                    with open(filepath, 'rb') as file:
                        content = file.read()
                        result.append({
                            "type": "video",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(content).decode('utf-8')
                            }
                        })
                else:
                    with open(filepath, 'r') as file:
                        content = file.read()
                        result.append({
                            "type": "text",
                            "text": f"```\n{content}\n```"
                        })
            except (IOError, OSError) as e:
                self.nvim.err_write(
                    f"Error reading file {filepath}: {e}\n")
                result.append({
                    "type": "text",
                    "text": input_string[match.start():match.end()]
                })

            last_end = match.end()

        # Add any remaining text after last tag
        if last_end < len(input_string):
            result.append({
                "type": "text",
                "text": input_string[last_end:]
            })

        return result

    def _truncate_conversation(self, messages: List[dict], max_tokens: int) -> List[dict]:
        if not messages or len(messages) == 1:
            return messages

        prompt_tokens, message_tokens = self._count_tokens(messages)

        total_tokens = prompt_tokens + \
            sum([m.input_tokens for m in message_tokens])

        if total_tokens <= max_tokens:
            return messages

        truncated_messages = []
        current_tokens = prompt_tokens

        # Iterate over the messages in reverse order
        for i in range(len(messages) - 1, 0, -1):
            if current_tokens + message_tokens[i].input_tokens <= max_tokens:
                truncated_messages.insert(0, messages[i])
                current_tokens += message_tokens[i].input_tokens
            else:
                break

        # If we pared down to no messages at all, keep the last message
        if not truncated_messages:
            truncated_messages = messages[-1:]
        return truncated_messages

    def _wrap_new_content_with_user_tags(self) -> None:
        """Wrap new content in <user></user> tags and save it back to the buffer."""
        buffer_content = '\n'.join(self.nvim.current.buffer[:])
        if (buffer_content.strip().startswith("<user>") and
                buffer_content.strip().endswith("</user>")):
            return
        has_previous_response = '</assistant>' in buffer_content
        if has_previous_response:
            # Get only the new content after the last </assistant> tag
            last_response_end = buffer_content.rindex(
                '</assistant>') + len('</assistant>')
            new_content = buffer_content[last_response_end:].strip()
            # Only wrap if there is actual content after </assistant>
            if new_content:
                wrapped_content = f"\n\n<user>\n{new_content}\n</user>"
                # Preserve the content before the new content and append the wrapped content
                updated_content = buffer_content[:last_response_end] + \
                    wrapped_content
                self.nvim.current.buffer[:] = updated_content.split('\n')
            else:
                # No new content to wrap, keep buffer as-is
                return
        else:
            new_content = buffer_content.strip()
            if new_content:
                wrapped_content = f"<user>\n{new_content}\n</user>"
                self.nvim.current.buffer[:] = wrapped_content.split('\n')

    def do_completion(self, completion_method: Callable, args: List[str]) -> None:
        self._wrap_new_content_with_user_tags()
        buffer_content = '\n'.join(self.nvim.current.buffer[:])
        buffer_number = self.nvim.current.buffer.number

        # Check for :b<buffer_number> pattern with at least one surrounding
        # whitespace and replace with buffer content
        def replace_buffer(match):
            buffer_number = int(match.group(1))
            if buffer_number in [b.number for b in self.nvim.buffers]:
                return '\n'.join(self.nvim.buffers[buffer_number][:])
            else:
                self.nvim.out_write(f"Buffer not found: {buffer_number}\n")
                return match.group(0)

        buffer_content = BUFFER_PATTERN.sub(replace_buffer, buffer_content)

        messages = []
        # Pattern to match both <user> and <assistant> tags
        pattern = re.compile(r'<(user|assistant)>([\s\S]*?)</\1>')
        for match in pattern.finditer(buffer_content):
            role = match.group(1)
            content = match.group(2).strip()
            if role == "user":
                content = self._process_content(content)
            messages.append({
                "role": role,
                "content": content
            })

        if self.truncate_conversation:
            truncated_messages = self._truncate_conversation(
                messages, self.max_context_tokens)
            if len(truncated_messages) < len(messages):
                self.nvim.out_write(
                    f"Truncated conversation from {len(messages)} to "
                    f"{len(truncated_messages)} messages.\n")
                messages = truncated_messages
        try:
            response_stream = completion_method(
                system=self.system_prompt,
                model=self.current_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=Timeout(10.0, read=self.timeout, write=self.timeout),
                stream=True
            )

            self.nvim.current.buffer.append(["", "<assistant>", ""])
            response_text = ""
            for chunk in response_stream:
                if chunk.type == "content_block_delta":
                    response_text += chunk.delta.text
                    lines = chunk.delta.text.split('\n')
                    self.nvim.current.buffer[-1] += lines[0]
                    if len(lines) > 1:
                        self.nvim.current.buffer.append(lines[1:])
                    # Move cursor to the last line of the buffer after appending each chunk
                    self.nvim.current.window.cursor = (
                        len(self.nvim.current.buffer), 0)
            self.nvim.current.buffer.append(["", "</assistant>", "", ""])

            buffer_number = self.nvim.current.buffer.number
            if self.buffer_filenames.get(buffer_number):
                # Make sure filename is sanitized
                sanitized_filename = shlex.quote(
                    self.buffer_filenames[buffer_number])
            else:
                # Get a filename if we don't already have one by asking Claude
                filename_prompt = """
                    Based on our conversation, suggest a descriptive filename
                    for saving this chat. Respond with only the filename,
                    no explanation. Keep the filename pithy and descriptive.
                    Extention should be .txt.
                """
                filename_response = completion_method(
                    model=self.filename_model,
                    messages=messages + [
                        {"role": "assistant", "content": response_text},
                        {"role": "user", "content": filename_prompt},
                    ],
                    max_tokens=100
                )

                filename = self._get_filename_from_response(filename_response)
                # Escape spaces and special characters in filename
                sanitized_filename = shlex.quote(filename)
                sanitized_filename = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S_") + sanitized_filename
            self.buffer_filenames[buffer_number] = sanitized_filename
            self.nvim.out_write(f"Saving to {sanitized_filename}\n")
            cmd = f'write! {sanitized_filename}'
            self.nvim.command(cmd)

            # Move cursor to the last line of the buffer after appending response
            self.nvim.current.window.cursor = (
                len(self.nvim.current.buffer), 0)

        except Exception as e:
            self.nvim.err_write(
                f"Error generating response:\n{format_exc()}\n")

    def get_claude_models(self) -> List[str]:
        """Get a list of available Anthropic models."""
        models = self.claude_client.models.list()
        return [model.id for model in models.data]

    @pynvim.command('Cl', nargs='0', sync=False)
    def claude_command(self, args: List[str]) -> None:
        return self.do_completion(self.claude_client.messages.create, args)

    @pynvim.command('Claude', nargs='0', sync=False)
    def claude_full_command(self, args: List[str]) -> None:
        return self.claude_command(args)

    @pynvim.command('Completion', nargs='0', sync=False)
    def claude_full_command(self, args: List[str]) -> None:
        return self.claude_command(args)

    @pynvim.command('CM', nargs='?', sync=True)
    def claude_model_command(self, args: List[str]) -> None:
        """Change the current model to the one provided or list available
        models if no argument is given."""
        if args:
            models = self.get_claude_models()
            model_name_or_index = args[0]
            try:
                model_index = int(model_name_or_index) - 1
                if model_index < 0 or model_index >= len(models):
                    self.nvim.err_write(
                        f"Error: {model_name_or_index} is not a valid model index.\n")
                    return
                model = models[model_index]
            except (ValueError, TypeError):
                if model_name_or_index not in models:
                    self.nvim.err_write(
                        f"Error: {model_name_or_index} is not a valid model.\n")
                    return
                model = model_name_or_index
            self.current_model = model
            self.nvim.out_write(f"Current model changed to {model}.\n")
            self._save_attributes_to_file()
        else:
            # Format the model list for display
            model_list = ["Available Claude models:"]
            for i, model in enumerate(self.get_claude_models()):
                if model == self.current_model:
                    model_list.append(f"  {i + 1}. {model} (current)")
                else:
                    model_list.append(f"  {i + 1}. {model}")
            self.nvim.out_write("\n".join(model_list) + "\n")

    @pynvim.command('ClaudeModel', nargs='?', sync=True)
    def claude_model_full_command(self, args: List[str]) -> None:
        return self.claude_model_command(args)

    @pynvim.command('BC', nargs='0', sync=True)
    def buffer_code_command(self, args: List[str]) -> None:
        buffer_content = '\n'.join(self.nvim.current.buffer[:])

        last_response, start_line = self._find_last_response(buffer_content)
        if last_response is None:
            self.nvim.err_write("No response found in buffer\n")
            return

        code_blocks = self._extract_code_blocks(last_response, start_line)

        original_buffer_number = self.nvim.current.buffer.number

        for _, _, code, _ in code_blocks:
            try:
                # Create a new buffer and set its content to the code block
                self.nvim.command('enew')
                self.nvim.current.buffer[:] = code.split('\n')
                self.nvim.out_write(
                    "Opened new buffer with number "
                    f"{self.nvim.current.buffer.number}\n")
            except Exception as e:
                self.nvim.err_write(
                    f"Error opening new buffer for code block:\n{format_exc()}\n"
                )

        # Return to the original buffer
        self.nvim.command(f'buffer {original_buffer_number}')

    @pynvim.command('BufferCode', nargs='0', sync=True)
    def buffer_code_full_command(self, args: List[str]) -> None:
        return self.buffer_code_command(args)

    @pynvim.command('WC', nargs='0', sync=True)
    def write_code_command(self, args: List[str]) -> None:
        """Write code blocks to files and update the buffer with filenames."""
        buffer_content = '\n'.join(self.nvim.current.buffer[:])

        last_response, start_line = self._find_last_response(buffer_content)
        if last_response is None:
            self.nvim.err_write("No response found in buffer\n")
            return

        code_blocks = self._extract_code_blocks(last_response, start_line)
        buffer = self.nvim.current.buffer

        for language, filename, code, line_number in code_blocks:
            # Generate filename if none exists
            if not filename:
                filename = self._generate_filename(language, filename, code)

            try:
                # Find the code block in the buffer
                for i in range(line_number, len(buffer)):
                    # Match either ```language or just ``` for no language
                    if buffer[i] == '```' or buffer[i] == f'```{language}':
                        # Get the comment style for this language
                        comment_style = self.COMMENT_STYLES.get(
                            language.lower(), '#')

                        # Check if next line is a shebang
                        next_line = buffer[i + 1] if i + \
                            1 < len(buffer) else ''
                        filename_line = f'{comment_style} {filename}'

                        if next_line.startswith('#!'):
                            # Check the line after shebang
                            if i + 2 >= len(buffer) or buffer[i + 2] != filename_line:
                                buffer[i + 2:i + 2] = [filename_line]
                            # Include shebang in the saved file
                            code = next_line + '\n' + code
                        else:
                            # Check if filename is already in the next line
                            if i + 1 >= len(buffer) or buffer[i + 1] != filename_line:
                                buffer[i + 1:i + 1] = [filename_line]
                        break

                # Write the code to the file
                with open(filename, 'w') as f:
                    f.write(code)
                self.nvim.out_write(f"Saved code block to {filename}\n")
            except Exception as e:
                self.nvim.err_write(
                    f"Error saving code block:\n{format_exc()}\n"
                )

    @pynvim.command('WriteCode', nargs='0', sync=True)
    def write_code_full_command(self, args: List[str]) -> None:
        return self.write_code_command(args)

    @pynvim.command('CP', nargs='0', sync=True)
    def copy_prompt_command(self, args: List[str]) -> None:
        """Copy the system prompt to a new buffer."""
        try:
            # Create a new buffer and set its content to the system prompt
            self.nvim.command('enew')
            self.nvim.current.buffer[:] = (
                self.system_prompt
                .strip()
                .split('\n')
            )
            self.nvim.out_write(
                "Opened new buffer with system prompt, buffer number: "
                f"{self.nvim.current.buffer.number}\n")
        except Exception as e:
            self.nvim.err_write(
                f"Error opening new buffer for system prompt:\n{format_exc()}\n"
            )

    @pynvim.command('CopyPrompt', nargs='0', sync=True)
    def copy_prompt_full_command(self, args: List[str]) -> None:
        return self.copy_prompt_command(args)

    @pynvim.command('ReplacePrompt', nargs='?', sync=True)
    def replace_prompt_command(self, args: List[str]) -> None:
        """Replace the system prompt with the contents of the specified buffer."""
        try:
            if args:
                try:
                    buffer_number = int(args[0])
                    if buffer_number not in [b.number for b in self.nvim.buffers]:
                        self.nvim.err_write(
                            f"Error: Buffer {buffer_number} does not exist.\n")
                        return
                    buffer_content = '\n'.join(
                        self.nvim.buffers[buffer_number][:])
                except ValueError:
                    filename = args[0]
                    with open(filename, 'r') as f:
                        buffer_content = f.read()
            else:
                buffer_number = self.nvim.current.buffer.number
                buffer_content = '\n'.join(self.nvim.current.buffer[:])

            self.system_prompt = buffer_content
            self.nvim.out_write(
                f"System prompt replaced with contents of buffer {buffer_number}.\n")
            self._save_attributes_to_file()
        except Exception as e:
            self.nvim.err_write(
                f"Error replacing system prompt:\n{format_exc()}\n"
            )

    @pynvim.command('MT', nargs='?', sync=True)
    def max_tokens_command(self, args: List[str]) -> None:
        """Show or change the maximum number of tokens."""
        try:
            if args:
                new_max_tokens = int(args[0])
                if (new_max_tokens < ABSOLUTE_MIN_TOKENS or
                        new_max_tokens > ABSOLUTE_MAX_TOKENS):
                    self.nvim.err_write(
                        f"Error: max tokens must be between {ABSOLUTE_MIN_TOKENS} "
                        f"and {ABSOLUTE_MAX_TOKENS}.\n")
                    return
                self.max_tokens = new_max_tokens
                self.nvim.out_write(
                    f"Max tokens changed to {new_max_tokens}.\n")
                self._save_attributes_to_file()
            else:
                self.nvim.out_write(
                    f"Current max tokens: {self.max_tokens}\n")
        except ValueError:
            self.nvim.err_write(
                "Error: max tokens must be an integer.\n")
        except Exception as e:
            self.nvim.err_write(
                f"Error changing max tokens:\n{format_exc()}\n"
            )

    @pynvim.command('MaxTokens', nargs='?', sync=True)
    def max_tokens_full_command(self, args: List[str]) -> None:
        return self.max_tokens_command(args)

    @pynvim.command('TC', nargs='0', sync=True)
    def token_count_command(self, args: List[str]) -> None:
        """Respond with the number of tokens in the current buffer."""
        try:
            buffer_content = '\n'.join(self.nvim.current.buffer[:])
            messages = [{"role": "user", "content": buffer_content}]
            prompt_tokens, message_tokens = self._count_tokens(messages)
            total_tokens = prompt_tokens + \
                sum([m.input_tokens for m in message_tokens])
            self.nvim.out_write(
                f"Current buffer token count: {total_tokens}\n")
        except Exception as e:
            self.nvim.err_write(
                f"Error counting tokens in current buffer:\n{format_exc()}\n"
            )

    @pynvim.command('TokenCount', nargs='0', sync=True)
    def token_count_full_command(self, args: List[str]) -> None:
        return self.token_count_command(args)

    @pynvim.command('Tr', nargs='?', sync=True)
    def truncate_conversation_command(self, args: List[str]) -> None:
        """Toggle truncation of the conversation."""
        if args:
            self.truncate_conversation = args[0].lower() == 'on'
            self._save_attributes_to_file()
        self.nvim.out_write(
            "Truncation of the conversation is "
            f"{'' if self.truncate_conversation else 'not '}enabled.\n")

    @pynvim.command('Truncate', nargs='?', sync=True)
    def truncate_conversation_full_command(self, args: List[str]) -> None:
        return self.truncate_conversation_command(args)

    @pynvim.command('CS', nargs='?', sync=True)
    def load_claude_settings_command(self, args: List[str]) -> None:
        if not args:
            pass
        elif args[0].lower() == 'defaults':
            self.current_model = "claude-3-5-haiku-20241022"
            self.filename_model = "claude-3-5-haiku-20241022"
            self.max_tokens = DEFAULT_MAX_TOKENS
            self.truncate_conversation = True
            self.max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS
            self.temperature = DEFAULT_TEMPERATURE
            self.timeout = DEFAULT_TIMEOUT
            self._save_attributes_to_file()
            self.nvim.out_write("Settings restored to defaults.\n")
        elif args[0].lower() == 'save':
            self._save_attributes_to_file()
            self.nvim.out_write("Settings saved.\n")
        elif args[0].find('=') != -1:
            setting, value = [s.strip() for s in args[0].split('=')]
            if setting not in [
                'model', 'filename_model', 'max_tokens', 'max_context_tokens',
                'truncate', 'temperature', 'timeout'
            ]:
                self.nvim.err_write(
                    f"Error: {setting} is not a valid setting.\n")
                return
            if setting == 'model':
                value = value.strip()
                if value not in self.get_claude_models():
                    self.nvim.err_write(
                        f"Error: {value} is not a valid model.\n")
                    return
                self.current_model = value
            elif setting == 'filename_model':
                value = value.strip()
                if value not in self.get_claude_models():
                    self.nvim.err_write(
                        f"Error: {value} is not a valid model.\n")
                    return
                self.filename_model = value
            elif setting == 'max_tokens':
                value = int(value)
                if (value < ABSOLUTE_MIN_TOKENS or value > ABSOLUTE_MAX_TOKENS):
                    self.nvim.err_write(
                        "Error: max tokens must be between "
                        f"{ABSOLUTE_MIN_TOKENS} and {ABSOLUTE_MAX_TOKENS}.\n")
                    return
                self.max_tokens = value
            elif setting == 'max_context_tokens':
                value = int(value)
                if (value < ABSOLUTE_MIN_CONTEXT_TOKENS or
                        value > ABSOLUTE_MAX_CONTEXT_TOKENS):
                    self.nvim.err_write(
                        "Error: max context tokens must be between "
                        f"{ABSOLUTE_MIN_CONTEXT_TOKENS} and "
                        f"{ABSOLUTE_MAX_CONTEXT_TOKENS}.\n")
                    return
                self.max_context_tokens = value
            elif setting == 'truncate':
                value = value.strip()
                if value not in ['on', 'off']:
                    self.nvim.err_write(
                        "Error: truncate must be 'on' or 'off'.\n")
                    return
                self.truncate_conversation = value == 'on'
            elif setting == 'temperature':
                value = float(value)
                if value < 0.0 or value > 1.0:
                    self.nvim.err_write(
                        "Error: temperature must be between 0.0 and 1.0.\n")
                    return
                self.temperature = value
            elif setting == 'timeout':
                value = float(value)
                if value < 0.0 or value > 600.0:  # 0 to 10 minutes
                    self.nvim.err_write(
                        "Error: timeout must be between 0.0 and 600.0.\n")
                    return
                self.timeout = value
        else:
            self.nvim.err_write(
                "Invalid argument. Use 'defaults' or 'save' or "
                "'model=...', 'filename_model=...', 'max_tokens=...', "
                "'max_context_tokens=...', 'truncate=...', 'temperature=...', "
                "'timeout=...'\n")
            return
        self.nvim.out_write(
            f"Current settings:\n"
            f"model: {self.current_model}\n"
            f"filename_model: {self.filename_model}\n"
            f"max_tokens: {self.max_tokens}\n"
            f"max_context_tokens: {self.max_context_tokens}\n"
            f"truncate_conversation: {self.truncate_conversation}\n"
            f"temperature: {self.temperature}\n"
            f"timeout: {self.timeout}\n"
        )

    @pynvim.command('ClaudeSettings', nargs='?', sync=True)
    def claude_settings_full_command(self, args: List[str]) -> None:
        return self.load_claude_settings_command(args)
