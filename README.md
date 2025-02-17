# Neovim AI Plugin

## Overview

The Neovim AI Plugin is a powerful integration that allows seamless interaction
with AI models directly within Neovim. Supporting multiple AI models through
OpenRouter's API, including Anthropic's Claude, it provides advanced features for
code generation, conversation management, and AI-assisted workflow.

## Prerequisites

- Neovim
- Python 3.8+
- `pynvim` package
- `httpx` package
- `anthropic` package
- OpenRouter API key (primary)
- Anthropic API key (optional, for token counting)

## Installation

Run the `setup.sh` script to install the dependencies and add the plugin to your
Neovim configuration.

```bash
cd <repository directory>
bash setup.sh
export OPENROUTER_API_KEY=<your-api-key>
```

OR

1. Install dependencies:
```bash
pip install pynvim httpx anthropic
export OPENROUTER_API_KEY=<your-api-key>
```

2. Add the plugin to your Neovim configuration

Following installation, run nvim, type `:UpdateRemotePlugin` and press <enter>,
close nvim, and then reopen it.

Add your OpenRouter API key to your `.bashrc` or `.zshrc` if you want to use the
plugin without setting the `OPENROUTER_API_KEY` environment variable.

## Commands

### Conversation Management

- `:Ai` (`:AI`, `:Completion`): Send current conversation to AI
- `:Models` (`:PM`, `:PrintModel`): List or select available models
- `:AISettings`: Load, save, or modify AI settings

### Code Interaction

- `:WC` (`:WriteCode`): Extract and save code blocks from last response
- `:BC` (`:BufferCode`): Open code blocks in new buffers

### Prompt Management

- `:CP` (`:CopyPrompt`): Copy current system prompt to a new buffer
- `:ReplacePrompt`: Replace system prompt with buffer contents

### Token Management

- `:MT` (`:MaxTokens`): Show or change the maximum number of tokens
- `:TC` (`:TokenCount`): Respond with the number of tokens in the current buffer

### Conversation Truncation

- `:Tr` (`:Truncate`): Toggle truncation of the conversation

## Buffer Reference

Reference other buffers using `:b<number>` syntax when talking to the AI.
This will pull in the contents of the buffer into the conversation before
sending it to the AI model.

## System Prompt

The plugin uses a configurable system prompt that guides the AI's responses,
focusing on code quality, formatting, and best practices.

## Configuration

Customize behavior using the `:AISettings` command with the following options:
- `model`: Set the AI model to use
- `filename_model`: Set the model used for filename generation
- `max_tokens`: Set maximum tokens for responses (128-8192)
- `max_context_tokens`: Set maximum context tokens (1024-204800)
- `truncate`: Toggle conversation truncation ('on'/'off')
- `temperature`: Set temperature for responses (0.0-1.0)
- `timeout`: Set request timeout in seconds (0.0-600.0)
- `limit_window`: Set rate limit window in seconds (0.0-600.0)

Examples:
```
:AISettings model=anthropic/claude-3.5-haiku-20241022
:AISettings max_tokens=4096
:AISettings defaults    # Reset to defaults
:AISettings save       # Save current settings
```

## Content Tags

You can include file contents in your prompts using the `<content>` tag:
```
<content>/path/to/file.txt</content>
```
The plugin supports both text files and media files (images, videos).

## Example Workflow

1. Open Neovim
2. Write a prompt and use `:Ai` to send it to the AI
3. Use `:WC` to save any generated code to files in the current directory
4. Use `:BC` to open code blocks in new buffers
5. Modify and iterate

The plugin will automatically wrap new content with `<user>` tags and will
save the conversation to a file in the current directory with a name suggested
by the AI.

It saves after every round, except that limitations within Neovim
will prevent the last response from being saved, so use `:w` to
save the file after the last round.

## Things to Try

You can continue any existing conversation by opening its corresponding file.

If you don't like a response, you can delete it, modify your
last prompt, and send it again. You can also modify the AI's side of the
conversation anywhere in the file. This will affect how the AI understands
the conversation history.

Use `:CP` to copy the system prompt to a new buffer, modify it, and then
use `:ReplacePrompt <buffer number>` to replace the system prompt with the new contents.

## Testing

Run
```
python -m unittest discover -s tests
```

## License

Neovim AI Plugin - AI integration for Neovim
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

## Contributing

[Contribution Guidelines]
