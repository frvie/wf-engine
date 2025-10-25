# 🚀 Workflow Engine

A high-performance, modular workflow orchestration engine for Python with automatic node discovery, environment isolation, and multi-backend inference support.


## 🎯 What is the Workflow Engine? 
High-performance, extensible workflow system for AI object detection with support for DirectML GPU, CUDA, NPU, and CPU backends. Features parallel execution, data session caching, and easy custom node creation.High-performance, extensible workflow system for AI object detection with support for DirectML GPU, CUDA, NPU, and CPU backends. Features parallel execution, data session caching, and easy custom node creation.

The Workflow Engine is a flexible task orchestration system that executes complex workflows by automatically discovering, managing, and executing modular Python nodes. It intelligently handles dependency conflicts through environment isolation and supports parallel execution with dependency resolution.

Originally built for multi-backend AI inference (DirectML GPU, Intel NPU, CPU), the engine is designed to be **general-purpose** and can orchestrate any type of computational workflow.## 🚀 Quick Start

## ✨ Main Capabilities

- 🔍 **Automatic Node Discovery** - Drop Python files in `workflow_nodes/` and they're automatically registered```powershell## 🚀 Quick Start

- 🔀 **Parallel Execution** - Executes nodes concurrently when dependencies allow

- 🔒 **Environment Isolation** - Runs nodes in isolated Python environments to handle conflicting dependencies# Run YOLOv8 detection across all backends

- 📊 **Dependency Resolution** - Builds and executes workflows based on dependency graphs

- 🎨 **Declarative Workflows** - Define workflows in simple JSON format

- 📦 **Automatic Dependency Management** - Installs required packages per node automatically# Run YOLOv8 detection across all backends

## 🛠️ Requirements 
This project requires **[uv](https://github.com/astral-sh/uv)** as the package manager.

## Create Virtual Environment
```bash
uv sync

## Run Demo Workflow

```bash

uv run python function_workflow_engine.py workflows/modular_function_based_demo.json

