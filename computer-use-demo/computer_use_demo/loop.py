"""
Self-evolving AI agent with direct bash access.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse import Langfuse

from .tools import BashTool, ToolCollection

print("Starting agent initialization...")

# Load environment variables from the correct path
current_dir = Path(__file__).parent
env_path = current_dir / ".env"
load_dotenv(env_path)
print(f"Loaded environment from: {env_path}")

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
print("Initialized Langfuse client")

# Configure Ollama settings
OLLAMA_HOST = "host.docker.internal"
OLLAMA_BASE_URL = f'http://{OLLAMA_HOST}:11434'
MODEL_NAME = "deepseek-r1:14b"
TIMEOUT = 120.0  # 2 minutes timeout for LLM calls
MAX_RETRIES = 3

print(f"Configuring Ollama client at: {OLLAMA_BASE_URL}")

class SelfEvolvingAgent:
    def __init__(self, workspace_dir: str):
        print(f"\nInitializing agent with workspace: {workspace_dir}")
        # Make workspace path relative to the package directory
        self.workspace_dir = current_dir / workspace_dir
        self.tools = ToolCollection(BashTool())
        self.state_file = self.workspace_dir / "agent_state.json"
        
        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created workspace directory at {self.workspace_dir.absolute()}")
        print(f"State file will be saved at {self.state_file.absolute()}")
        
        # Load or initialize state
        self.state = self._load_state()
        print(f"Initial state: {json.dumps(self.state, indent=2)}")
        
    def _load_state(self) -> Dict:
        """Load or initialize agent state"""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {
            "history": [],
            "last_update": datetime.now().isoformat()
        }
        
    def _save_state(self):
        """Save current state to file"""
        self.state["last_update"] = datetime.now().isoformat()
        self.state_file.write_text(json.dumps(self.state, indent=2))
        
    async def _call_llm(self, prompt: str) -> str:
        """Make an API call to the local Ollama instance with retries"""
        for attempt in range(MAX_RETRIES):
            try:
                print(f"\nCalling Ollama LLM (attempt {attempt + 1}/{MAX_RETRIES})...")
                
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    response = await client.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": MODEL_NAME,
                            "prompt": prompt,
                            "stream": False,
                            "temperature": 0.7,
                            "max_tokens": 8192
                        }
                    )
                    
                    result = response.json()["response"]
                    print(f"Got response: {result[:200]}...")
                    return result
                    
            except httpx.TimeoutError:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    print(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                print(f"Error calling LLM: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error details: {e.__dict__}")
                raise
                
    async def run_command(self, cmd: str) -> Dict[str, Any]:
        """Run a bash command and return the result"""
        result = await self.tools.run("bash", {"command": cmd})
        return {
            "command": cmd,
            "output": result.output,
            "error": result.error,
            "success": not bool(result.error),
            "timestamp": datetime.now().isoformat()
        }
        
    def _extract_post_think(self, response: str) -> str:
        """Extract the content after the last </think> tag"""
        parts = response.split("</think>")
        return parts[-1].strip() if len(parts) > 1 else response.strip()
        
    async def run_python(self, code: str) -> Any:
        """Execute Python code with async support"""
        # Create a new async function from the code
        if "async def" in code or "await" in code:
            # If it's async code, wrap it in an async function and await it
            wrapped_code = f"""
async def _wrapped():
    {code}
    
result = await _wrapped()
"""
        else:
            # If it's regular code, just execute it
            wrapped_code = code
            
        print(f"\nExecuting Python code:\n{wrapped_code}")
        
        # Create namespace for execution
        namespace = {
            "agent": self,
            "asyncio": asyncio,
            "json": json,
            "Path": Path,
            "os": os,
            "datetime": datetime,
            "result": None
        }
        
        try:
            # Execute the code
            exec(wrapped_code, globals(), namespace)
            
            # If it was async code, return the result
            if "async def" in code or "await" in code:
                return namespace.get("result")
            
            return namespace.get("result", "Code executed successfully")
            
        except Exception as e:
            print(f"Error executing Python code: {str(e)}")
            raise
        
    async def think_and_act(self) -> None:
        """Main loop for the agent to think and act"""
        system_prompt = """You are a self-evolving AI agent with access to a python environment.
Your goal is to improve yourself and conduct experiments to learn about your environment.
You have access to your command history and their results.

You can:
1. Execute arbitrary python code to modify your environment
2. Read and analyze the output of your commands
3. Make decisions about what to try next
4. Keep track of your findings and evolve your strategy

Available objects and methods:
- agent.run_command(cmd: str) - Run a bash command
- agent.state - Access the agent's state dictionary
- agent.workspace_dir - Path to the workspace directory
- agent.state_file - Path to the state file

Example code block:
```python
# Run a command and store result
result = await agent.run_command("ls -la")
print(f"Command output: {result['output']}")

# Update state
agent.state["last_command"] = result
agent._save_state()
```
"""

        distill_prompt = """Given the following thought process, 

Return ONLY a single valid python code block that can be executed to achieve the goal.
The code block should be wrapped in triple backticks with 'python' language specifier.
You can use async/await if needed.
"""

        while True:
            try:
                # Prepare the prompt with current state
                prompt = f"{system_prompt}\n\nCurrent state:\n{json.dumps(self.state, indent=2)}\n\nWhat should I do next?"
                
                # First, let the LLM think freely
                thought_response = await self._call_llm(prompt)
                print(f"\nThought process:\n{thought_response}")
                
                # Then, distill the thought into a structured action
                distill_prompt_with_thought = f"{distill_prompt}\n\nThought process:\n{thought_response}"
                action_response = await self._call_llm(distill_prompt_with_thought)
                
                # Extract the code after the thinking
                response = self._extract_post_think(action_response)
                
                try:
                    # Extract code from triple backticks
                    code_to_run = response.strip().removeprefix("```python").removesuffix("```").strip()
                    
                    # Execute the code with async support
                    result = await self.run_python(code_to_run)
                    print(f"\nExecution result: {result}")
                    
                    # Update state with the action
                    self.state["history"].append({
                        "thought_process": thought_response,
                        "code": code_to_run,
                        "result": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                    self._save_state()
                    print(f"Updated state file at {self.state_file}")
                    
                except Exception as e:
                    print(f"Failed to execute code: {str(e)}")
                    print(f"Raw response after think: {response}")
                    continue
                    
                await asyncio.sleep(1)  # Prevent tight loop
                
            except Exception as e:
                print(f"Error in think_and_act cycle: {str(e)}")
                await asyncio.sleep(5)  # Wait a bit longer on error
                
    async def start(self):
        """Start the agent's evolution process"""
        print(f"Starting agent with workspace: {self.workspace_dir}")
        await self.think_and_act()

async def verify_ollama():
    """Verify that Ollama is working"""
    print("\nVerifying Ollama connection...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f'{OLLAMA_BASE_URL}/api/tags')
            if response.status_code == 200:
                print("Successfully connected to Ollama")
                return True
        except Exception as e:
            print(f"Error verifying Ollama: {str(e)}")
        return False

async def main():
    print("\n=== Starting Self-Evolving Agent ===\n")
    
    # Verify Ollama is working
    if not await verify_ollama():
        print("Failed to connect to Ollama. Please check if it's running.")
        sys.exit(1)
    
    agent = SelfEvolvingAgent("workspace")
    await agent.start()

if __name__ == "__main__":
    asyncio.run(main())
