"""
Claude Neovim Plugin - AI integration for Neovim
Copyright (C) [2024] [P. Scott DeVos]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import json
import pynvim
import anthropic
import base64
import mimetypes
import os
import re
import shlex
from datetime import datetime
from typing import Optional, List, Tuple
from traceback import format_exc


# Pattern to match :b<buffer_number> with at least one surrounding whitespace
BUFFER_PATTERN = re.compile(r'\s+:b(\d+)\s+')
# Pattern to match material between <content> tags
CONTENT_PATTERN = re.compile(r'<content>(.*?)</content>', re.DOTALL)

# Maximum number of tokens Claude will return
DEFAULT_MAX_TOKENS = 4096
ABSOLUTE_MAX_TOKENS = 8192
ABSOLUTE_MIN_TOKENS = 512

DEFAULT_TEMPERATURE = 0.25

# Maximum number of tokens to send to Claude
ABSOLUTE_MIN_CONTEXT_TOKENS = 1024 * 200
MAX_CONTEXT_TOKENS = ABSOLUTE_MIN_CONTEXT_TOKENS

SETTINGS_FILE = 'nvim_claude.json'

DEFAULT_SYSTEM_PROMPT = """
    Please format all responses with line breaks at 80 columns.
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
"""


@pynvim.plugin
class ClaudePlugin:
    def __init__(self, nvim):
        self.nvim = nvim
        self.client = anthropic.Client(timeout=10.0)

        # Default settings
        default_config = {
            "current_model": "claude-3-5-haiku-20241022",
            "current_filename": None,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "truncate_conversation": True,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "temperature": DEFAULT_TEMPERATURE,
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
        self.current_filename = config["current_filename"]
        self.max_tokens = config["max_tokens"]
        self.max_context_tokens = config["max_context_tokens"]
        self.truncate_conversation = config["truncate_conversation"]
        self.system_prompt = config["system_prompt"]
        self.temperature = config["temperature"]

    def _save_attributes_to_file(self):
        attributes = {
            "current_model": self.current_model,
            "current_filename": self.current_filename,
            "max_tokens": self.max_tokens,
            "max_context_tokens": self.max_context_tokens,
            "truncate_conversation": self.truncate_conversation,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
        }
        config_dir = os.path.join(os.path.expanduser('~'), '.config', 'nvim')
        os.makedirs(config_dir, exist_ok=True)
        attributes_path = os.path.join(config_dir, SETTINGS_FILE)
        with open(attributes_path, 'w') as file:
            json.dump(attributes, file, indent=2)

    @staticmethod
    def _extract_code_blocks(text: str) -> List[Tuple[str, str]]:
        """Extracts code blocks from the given text."""
        code_blocks = re.finditer(r'```(\w*)\n([\s\S]*?)```', text)
        return [(block.group(1) or 'txt', block.group(2)) for block in code_blocks]

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

        return f"\n<assistant>\n{text}\n</assistant>\n\n"

    @staticmethod
    def _get_filename_from_response(response: anthropic.types.Message) -> str:
        """Get a filename from the response."""
        filename = (
            ClaudePlugin._format_response(response.content)
            .replace('<assistant>', '')
            .replace('</assistant>', '')
            .replace('\n', '')
            .strip()
        )
        # Escape spaces and special characters in filename
        return shlex.quote(filename)

    @staticmethod
    def get_claude_models() -> List[str]:
        """Get a list of available Anthropic models."""
        models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240229",
        ]
        return models

    def _count_tokens(self, messages: List[dict]) -> Tuple[int, int]:
        # Tokenize the system prompt and first message because we have to
        # include a message or anthropic will throw an error
        system_prompt_tokens_and_first_message = \
            self.client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=self.current_model,
                system=self.system_prompt,
                messages=[messages[0]],
            )
        # Tokenize each message and store the token counts
        message_tokens = [
            self.client.beta.messages.count_tokens(
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

    def _find_last_response(self, buffer_content: str) -> Optional[str]:
        cursor_pos = self.nvim.current.window.cursor[0] - 1
        buffer_lines = buffer_content.split('\n')

        # Check if the cursor is within or on an <assistant> tag
        for i in range(cursor_pos, -1, -1):
            line = buffer_lines[i]
            if '<assistant>' in line and '</assistant>' in line:
                start = line.find('<assistant>') + len('<assistant>')
                end = line.find('</assistant>')
                return line[start:end].strip()
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
                            .strip()
                        )

        # If no match is found, write an error
        self.nvim.err_write(
            "No <assistant> tag found around or above the cursor\n")
        return None

    def _generate_filename(self, language: str, code: str) -> str:
        filename_prompt = (
            f"Suggest an appropriate filename for this code block.\n"
            f"{language}:\n{code}\n\n"
            "Respond with only the filename, no explanation."
        )
        filename_response = self.client.messages.create(
            model=self.current_model,
            messages=[{"role": "user", "content": filename_prompt}],
            max_tokens=100
        )
        return self._get_filename_from_response(filename_response)

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
                with open(filepath, 'rb') as file:
                    content = file.read()
                    media_type, _ = mimetypes.guess_type(filepath)

                    result.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type or "application/octet-stream",
                            "data": base64.b64encode(content).decode('utf-8')
                        }
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

    def _prepare_content(self, original_content: str) -> str:
        """Extract content blocks from the given text."""
        text = original_content
        result = []
        while text:
            match = CONTENT_PATTERN.match(text)
            if match:
                before, filename, after = match.groups()
                if before:
                    result.append({
                        "type": "text",
                        "text": before
                    })
                filename = filename.strip()
                if filename.startswith("~"):
                    filename = os.path.expanduser(filename)
                file_extension = os.path.splitext(filename)[1].lower()
                media_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp',
                }.get(file_extension, 'text/plain')
                if not media_type:
                    media_type = "text/plain"
                content_type = media_type.split('/')[0]
                try:
                    with open(filename, 'rb') as file:
                        encoded_data = base64.b64encode(
                            file.read()).decode('utf-8')
                    result.append({
                        "type": content_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_data
                        }
                    })
                except Exception as e:
                    self.nvim.err_write(
                        f"Error loading file {filename}: {str(e)}\n")
                text = after
            else:
                result.append({
                    "type": "text",
                    "text": text
                })
                break
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
            wrapped_content = f"\n\n<user>\n{new_content}\n</user>"
            # Preserve the content before the new content and append the wrapped content
            updated_content = buffer_content[:last_response_end] + \
                wrapped_content
            self.nvim.current.buffer[:] = updated_content.split('\n')
        else:
            new_content = buffer_content
            wrapped_content = f"<user>\n{new_content}\n</user>"
            self.nvim.current.buffer[:] = wrapped_content.split('\n')

    @pynvim.command('Cl', nargs='0', sync=True)
    def claude_command(self, args: List[str]) -> None:

        self._wrap_new_content_with_user_tags()
        buffer_content = '\n'.join(self.nvim.current.buffer[:])

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
        is_continuation = True if len(messages) > 1 else False

        if self.truncate_conversation:
            truncated_messages = self._truncate_conversation(
                messages, self.max_context_tokens)
            if len(truncated_messages) < len(messages):
                self.nvim.out_write(
                    f"Truncated conversation from {len(messages)} to "
                    f"{len(truncated_messages)} messages.\n")
                messages = truncated_messages

        try:
            response = self.client.messages.create(
                system=self.system_prompt,
                model=self.current_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            formatted_response = self._format_response(response.content)

            # Append response to buffer
            self.nvim.current.buffer.append(formatted_response.split('\n'))

            # Get a filename if this is the first response and save after
            # each response (turn)
            if not is_continuation and not self.current_filename:
                # Ask Claude for a filename
                filename_prompt = """
                    Based on our conversation, suggest a descriptive filename
                    for saving this chat. Respond with only the filename,
                    no explanation. Keep the filename pithy and descriptive.
                    Extention should be .txt.
                """
                filename_response = self.client.messages.create(
                    model=self.current_model,
                    messages=messages + [
                        {"role": "assistant", "content": response.content},
                        {"role": "user", "content": filename_prompt},
                    ],
                    max_tokens=100
                )

                self.current_filename = \
                    self._get_filename_from_response(filename_response)
                # Escape spaces and special characters in filename
                sanitized_filename = shlex.quote(self.current_filename)
                sanitized_filename = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S_") + sanitized_filename
                cmd = f'write! {sanitized_filename}'
                self.nvim.out_write(f"Saving to {cmd}\n")
                self.nvim.command(cmd)
            elif self.current_filename:
                self.nvim.command('write')

            # Move cursor to the last line of the buffer after appending response
            self.nvim.current.window.cursor = (
                len(self.nvim.current.buffer), 0)

        except Exception as e:
            self.nvim.err_write(
                f"Error generating response:\n{format_exc()}\n")

    @pynvim.command('Claude', nargs='0', sync=True)
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

    @pynvim.command('Bc', nargs='0', sync=True)
    def buffer_code_command(self, args: List[str]) -> None:
        buffer_content = '\n'.join(self.nvim.current.buffer[:])

        last_response = self._find_last_response(buffer_content)
        if last_response is None:
            self.nvim.err_write("No response found in buffer\n")
            return

        code_blocks = self._extract_code_blocks(last_response)

        original_buffer_number = self.nvim.current.buffer.number

        for i, (language, code) in enumerate(code_blocks):
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

    @pynvim.command('Wc', nargs='0', sync=True)
    def write_code_command(self, args: List[str]) -> None:
        buffer_content = '\n'.join(self.nvim.current.buffer[:])

        last_response = self._find_last_response(buffer_content)
        if last_response is None:
            self.nvim.err_write("No response found in buffer\n")
            return

        code_blocks = self._extract_code_blocks(last_response)

        for i, (language, code) in enumerate(code_blocks):
            filename = self._generate_filename(language, code)

            try:
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

    @pynvim.command('Cp', nargs='0', sync=True)
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

    @pynvim.command('Rp', nargs='1', sync=True)
    def replace_prompt_command(self, args: List[str]) -> None:
        """Replace the system prompt with the contents of the specified buffer."""
        try:
            buffer_number = int(args[0])
            if buffer_number not in [b.number for b in self.nvim.buffers]:
                self.nvim.err_write(
                    f"Error: Buffer {buffer_number} does not exist.\n")
                return

            buffer_content = '\n'.join(self.nvim.buffers[buffer_number][:])
            self.system_prompt = buffer_content
            self.nvim.out_write(
                f"System prompt replaced with contents of buffer {buffer_number}.\n")
            self._save_attributes_to_file()
        except Exception as e:
            self.nvim.err_write(
                f"Error replacing system prompt:\n{format_exc()}\n"
            )

    @pynvim.command('ReplacePrompt', nargs='1', sync=True)
    def replace_prompt_full_command(self, args: List[str]) -> None:
        return self.replace_prompt_command(args)

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
            self.max_tokens = DEFAULT_MAX_TOKENS
            self.truncate_conversation = True
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
            self._save_attributes_to_file()
            self.nvim.out_write("Settings restored to defaults.\n")
        elif args[0].lower() == 'save':
            self._save_attributes_to_file()
            self.nvim.out_write("Settings saved.\n")
        elif args[0].find('=') != -1:
            setting, value = [s.strip() for s in args[0].split('=')]
            if setting not in ['model', 'max_tokens', 'truncate', 'temperature']:
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
            elif setting == 'max_tokens':
                value = int(value)
                if (value < ABSOLUTE_MIN_TOKENS or value > ABSOLUTE_MAX_TOKENS):
                    self.nvim.err_write(
                        "Error: max tokens must be between "
                        f"{ABSOLUTE_MIN_TOKENS} and {ABSOLUTE_MAX_TOKENS}.\n")
                    return
                self.max_tokens = value
            elif setting == 'truncate':
                value = value.strip()
                if value not in ['on', 'off']:
                    self.nvim.err_write(
                        "Error: truncate must be 'on' or 'off'.\n")
                    return
                self.truncate_conversation = value == 'on'
        else:
            self.nvim.err_write(
                "Invalid argument. Use 'defaults' or 'save' or "
                "'model=...', 'max_tokens=...', 'truncate=...'\n")
            return
        self.nvim.out_write(
            f"Current settings:\n"
            f"model: {self.current_model}\n"
            f"max_tokens: {self.max_tokens}\n"
            f"max_context_tokens: {self.max_context_tokens}\n"
            f"truncate_conversation: {self.truncate_conversation}\n"
            f"temperature: {self.temperature}\n"
            f"system_prompt: {self.system_prompt}\n"
        )

    @pynvim.command('ClaudeSettings', nargs='?', sync=True)
    def claude_settings_full_command(self, args: List[str]) -> None:
        return self.load_claude_settings_command(args)
