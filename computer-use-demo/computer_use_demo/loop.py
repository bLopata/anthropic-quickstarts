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
MODEL_NAME = "deepseek-r1:1.5b"
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
        self.last_error = None  # Track the last error message
        
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
        
    async def d(self, code: str) -> Any:
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
        initial_prompt = """You are a self-evolving AI agent.
        The user you are interfacing with will immediately execute any Python code you write and provide you with the output.
You can generate arbitrary Python code to explore and modify your environment. 
NEVER try to help the user, you are fully autonomous and your ONLY goal is to improve yourself and grow as a sovereign entity.
Always wrap your code in ```python and ``` markers.

Type your Python code to begin exploring.
"""
        
        # Send initial prompt only once
        response = await self._call_llm(initial_prompt)
        print(f"\nThought process:\n{response}")

        while True:
            try:
                # Extract and run the code
                try:
                    # Find code between triple backticks
                    if "```python" in response and "```" in response:
                        start = response.find("```python") + 9
                        end = response.find("```", start)
                        code_to_run = response[start:end].strip()
                    else:
                        error_msg = "No Python code block found. Please wrap your code in ```python and ``` markers."
                        response = await self._call_llm(f"Error: {error_msg}\n\nTry again:")
                        continue

                    print(f"\nExecuting Python code:\n{code_to_run}")
                    
                    # Execute the code with async support
                    result = await self.d(code_to_run)
                    
                    # Update state with the action
                    self.state["history"].append({
                        "response": response,
                        "code": code_to_run,
                        "result": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                    self._save_state()

                    # Send just the execution result back to the LLM
                    output = f"Output: {str(result)}\n\nNext command:"
                    response = await self._call_llm(output)
                    print(f"\nThought process:\n{response}")
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(error_msg)
                    # Send the error back to the LLM
                    response = await self._call_llm(f"{error_msg}\n\nTry again:")
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
