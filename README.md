# Claude Neovim Plugin

## Overview

The Claude Neovim Plugin is a powerful integration that allows seamless interaction
with Anthropic's Claude AI directly within Neovim. It provides advanced features
for code generation, conversation management, and AI-assisted workflow.

## Prerequisites

- Neovim
- Python 3.8+
- `pynvim` package
- `anthropic` package
- Anthropic API key

## Installation

Run the `setup.sh` script to install the dependencies and add the plugin to your
Neovim configuration.

```bash
cd <repository directory>
bash setup.sh
export ANTHROPIC_API_KEY=<your-api-key>
```

OR

1. Install dependencies:
```bash
pip install pynvim anthropic
export ANTHROPIC_API_KEY=<your-api-key>
```

2. Add the plugin to your Neovim configuration

Following installation, run nvim, type `:UpdateRemotePlugin` and press <enter>,
close nvim, and then reopen it.

Add your Anthropic API key to your `.bashrc` or `.zshrc` if you want to use the
plugin without setting the `ANTHROPIC_API_KEY` environment variable.

## Commands

### Conversation Management

- `:Claude` (`:Cl`): Send current conversation to Claude
- `:ClaudeModel` (`:Cm`): Select AI model
- `:ClaudeModels`: List available models

### Code Interaction

- `:WriteCode` (`:Wc`): Extract and save code blocks from last response
- `:BufferCode` (`:Bc`): Open code blocks in new buffers

### Prompt Management

- `:CopyPrompt` (`:Cp`): Copy current system prompt to a new buffer
- `:ReplacePrompt` (`:Rp`): Replace system prompt with buffer contents

### Token Management

- `:MaxTokens` (`:MT`): Show or change the maximum number of tokens
- `:TokenCount` (`:TC`): Respond with the number of tokens in the current buffer

### Conversation Truncation

- `:Truncate` (`:Tr`): Toggle truncation of the conversation

### Settings Management

- `:ClaudeSettings` (`:CS`): Load, save, or reset Claude settings

## Buffer Reference

Reference other buffers using `:b<number>` syntax when talking to Claude.
This will pull in the contents of the buffer into the conversation before
sending it to Claude.

## System Prompt

The plugin uses a configurable system prompt that guides Claude's responses,
focusing on code quality, formatting, and best practices.

## Configuration

Customize behavior by editing `system_prompt.txt` or using `:Cp` and `:Rp`
commands.

## Advanced Features

- Automatic filename generation for code blocks
- Conversation history tracking
- Multi-model support

## Example Workflow

1. Open Neovim.
2. Write a prompt and use `:Cl` to send it to Claude.
3. Use `:Wc` to save any generated code to a file in the current directory
   with a name suggested by Claude.
4. Use `:Bc` to open code in buffers
5. Modify and iterate

The plugin will automatically wrap new content with `<user>` tags and will
save the conversation to a file in the current directory with a name suggested
by Claude.

It saves after every round, except that limitations within Neovim
will prevent the last response from Claude from being saved, so use `:w` to
save the file after the last round.

## Things to Try

You can continue any existing conversation by opening its corresponding file
in claude.

If you don't like a response from Claude, you can delete it, modify your
last prompt, and send it again. You can also modify Claude's side of the
conversation anywhere in the file. This will "Gaslight" Claude into thinking
that it wrote the responses the way you changed it.

Use `:Cp` to copy the system prompt to a new buffer, modify it, and then
use `:Rp <buffer number>` to replace the system prompt with the new contents.
Try adding this to the end of the system prompt to see what happens:

```
End every response with "I feel pretty!"
```

## Testing

Run
```
python -m unittest discover -s tests
```


## License

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

## Contributing

[Contribution Guidelines]
