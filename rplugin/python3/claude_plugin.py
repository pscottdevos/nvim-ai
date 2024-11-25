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
import pynvim
import anthropic
import os
import re
import shlex
from datetime import datetime
from typing import Optional, List, Tuple
from traceback import format_exc


# Pattern to match :b<buffer_number> with at least one surrounding whitespace
BUFFER_PATTERN = re.compile(r'\s+:b(\d+)\s+')

# Maximum number of tokens to send to Claude
MAX_TOKENS = 4096


@pynvim.plugin
class ClaudePlugin:
    def __init__(self, nvim):
        self.nvim = nvim
        self.client = anthropic.Client()
        self.current_model = "claude-3-5-haiku-20241022"
        self.current_filename: Optional[str] = None
        try:
            script_dir = os.path.dirname(__file__)
            prompt_path = os.path.join(script_dir, 'system_prompt.txt')
            with open(prompt_path, 'r') as file:
                self.system_prompt = file.read()
        except FileNotFoundError:
            self.nvim.err_write(
                "system_prompt.txt not found. Using default system prompt.\n")
            self.system_prompt = """
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
        except Exception as e:
            self.nvim.err_write(
                f"Error loading system prompt from file:\n{format_exc()}\n")
            self.system_prompt = ""

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
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
        return models

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

    def _truncate_conversation(self, messages: List[dict], max_tokens: int) -> List[dict]:

        # Tokenize the system prompt
        system_prompt_tokens = anthropic.count_tokens(self.system_prompt)

        # Tokenize each message and store the token counts
        message_tokens = [
            anthropic.count_tokens(msg["content"]) for msg in messages]
        total_tokens = system_prompt_tokens + sum(message_tokens)

        if total_tokens <= max_tokens:
            return messages

        truncated_messages = []
        current_tokens = system_prompt_tokens

        # Iterate over the messages in reverse order
        for i in range(len(messages) - 1, -1, -1):
            if current_tokens + message_tokens[i] <= max_tokens:
                truncated_messages.insert(0, messages[i])
                current_tokens += message_tokens[i]
            else:
                break

        return truncated_messages

    def _truncate_conversation(self, messages: List[dict], max_tokens: int) -> List[dict]:
        if not messages or len(messages) == 1:
            return messages
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

    @pynvim.command('CM', nargs='1', sync=True)
    @pynvim.command('ClaudeModel', nargs='1', sync=True)
    def claude_model_command(self, args: List[str]) -> None:
        """Change the current model to the one provided."""
        model = args[0]
        if model not in self.get_claude_models():
            self.nvim.err_write(f"Error: {model} is not a valid model.\n")
        else:
            self.current_model = model
            self.nvim.out_write(f"Current model changed to {model}.\n")

    @pynvim.command('ClaudeModels', nargs='0', sync=True)
    def claude_models_command(self, args: List[str]) -> None:
        """List available Anthropic models that can be used with :Claude command."""

        # Format the model list for display
        model_list = ["Available Claude models:"]
        for model in self.get_claude_models():
            if model == self.current_model:
                model_list.append(f"  - {model} (current)")
            else:
                model_list.append(f"  - {model}")

        self.nvim.out_write("\n".join(model_list) + "\n")

    @pynvim.command('Cl', nargs='0', sync=True)
    @pynvim.command('Claude', nargs='0', sync=True)
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
            messages.append({
                "role": role,
                "content": content
            })
        is_continuation = True if len(messages) > 1 else False

        messages = self._truncate_conversation(messages, MAX_TOKENS)

        try:
            response = self.client.messages.create(
                system=self.system_prompt,
                model=self.current_model,
                messages=messages,
                max_tokens=MAX_TOKENS
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

    @pynvim.command('Bc', nargs='0', sync=True)
    @pynvim.command('BufferCode', nargs='0', sync=True)
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

    @pynvim.command('Wc', nargs='0', sync=True)
    @pynvim.command('WriteCode', nargs='0', sync=True)
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

    @pynvim.command('Cp', nargs='0', sync=True)
    @pynvim.command('CopyPrompt', nargs='0', sync=True)
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

    @pynvim.command('Rp', nargs='1', sync=True)
    @pynvim.command('ReplacePrompt', nargs='1', sync=True)
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
        except Exception as e:
            self.nvim.err_write(
                f"Error replacing system prompt:\n{format_exc()}\n"
            )
