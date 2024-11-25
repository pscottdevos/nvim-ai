#!/bin/bash

# Install requirements from requirements.txt
python3 -m pip install -r requirements.txt

# Get the current directory
PLUGIN_DIR=$(dirname "$(realpath "$0")")

# Create required directories if they don't exist
mkdir -p ~/.config/nvim/rplugin/python3

# Create symlink
ln -sf "$PLUGIN_DIR/rplugin/python3/claude_plugin.py" ~/.config/nvim/rplugin/python3/claude_plugin.py

echo "Plugin installed. Please run :UpdateRemotePlugins in Neovim, then close and reopen nvim."