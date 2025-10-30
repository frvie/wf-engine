#!/usr/bin/env python
"""
MCP Server Launcher
Ensures proper stdio handling for Claude Desktop integration
"""
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add error logging
print("Starting MCP server...", file=sys.stderr)
print(f"Working directory: {os.getcwd()}", file=sys.stderr)
print(f"Python: {sys.executable}", file=sys.stderr)

# Import and run the MCP server
try:
    from src.interfaces import mcp_server
    import asyncio
    print("MCP server module loaded successfully", file=sys.stderr)
    asyncio.run(mcp_server.main())
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
