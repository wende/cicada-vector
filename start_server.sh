#!/bin/bash
# Helper to start the MCP server

# Ensure MCP is installed
if ! python3 -c "import mcp" 2>/dev/null; then
    echo "MCP not installed. Installing..."
    pip install ".[server]"
fi

# Set your Ollama settings if needed
export OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}
export OLLAMA_MODEL=${OLLAMA_MODEL:-nomic-embed-text}

# Point to your DB
export CICADA_HYBRID_DIR=${CICADA_HYBRID_DIR:-hybrid_csv_db}

echo "Starting Cicada Vector MCP Server..."
cicada-vec-server
