#!/usr/bin/env python3
"""
Workflow Engine CLI - Entry Point

This is a simple wrapper that calls the actual CLI implementation.
"""

if __name__ == "__main__":
    from src.interfaces.cli import main
    main()
