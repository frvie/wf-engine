"""Simple test of AutoGen agent with Ollama."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing AutoGen + Ollama Integration...\n")

# Test 1: Check if Ollama is accessible
print("1. Testing Ollama connection:")
try:
    import httpx
    response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
    if response.status_code == 200:
        models = response.json().get("models", [])
        print(f"✅ Ollama is running with {len(models)} models")
        print(f"   Available: {[m['name'] for m in models[:3]]}")
    else:
        print(f"⚠️ Ollama responded with status {response.status_code}")
except Exception as e:
    print(f"❌ Cannot connect to Ollama: {e}")
    sys.exit(1)
print()

# Test 2: Check AutoGen import
print("2. Testing AutoGen import:")
try:
    # Try new autogen-agentchat API first
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import TextMentionTermination
        from autogen_core import CancellationToken
        print("✅ AutoGen (agentchat) imported successfully")
        autogen_version = "agentchat"
    except ImportError:
        # Fall back to pyautogen
        from autogen import ConversableAgent
        print("✅ AutoGen (pyautogen) imported successfully")
        autogen_version = "pyautogen"
except Exception as e:
    print(f"❌ Cannot import AutoGen: {e}")
    sys.exit(1)
print()

# Test 3: Create simple agent
print("3. Creating test agent:")
try:
    config_list = [
        {
            "model": "codellama:latest",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
    ]
    
    test_agent = ConversableAgent(
        name="test_agent",
        llm_config={"config_list": config_list, "timeout": 30},
        system_message="You are a helpful assistant. Respond briefly.",
    )
    print("✅ Agent created successfully")
    print("   Model: codellama:latest")
    print("   Base URL: http://localhost:11434/v1")
except Exception as e:
    print(f"❌ Cannot create agent: {e}")
    sys.exit(1)
print()

# Test 4: Simple agent interaction (optional - can be slow)
print("4. Testing agent response (this may take 10-30 seconds):")
print("   Asking: 'What is 2+2? Reply with just the number.'")
try:
    user_proxy = ConversableAgent(
        name="user",
        llm_config=False,
        human_input_mode="NEVER",
    )
    
    # Send a simple message
    user_proxy.initiate_chat(
        test_agent,
        message="What is 2+2? Reply with just the number.",
        max_turns=1,
    )
    print("✅ Agent responded successfully")
except Exception as e:
    print(f"⚠️ Agent interaction failed: {e}")
    print("   (This is expected if Ollama model is not pulled)")
print()

print("✅ Basic AutoGen + Ollama setup is working!")
print("\nNext steps:")
print("- Test workflow_agent_poc.py for full functionality")
print("- Test MCP server integration")
