@echo off
REM Workflow Engine CLI Wrapper
REM This batch file makes 'wfe' available as a direct command

cd /d "%~dp0"
uv run python wfe.py %*