import asyncio
import os
import textwrap
import json
from typing import List, Optional
from openai import OpenAI
from core_rl.client import CoreRlEnv as EnvClient
from core_rl.models import Action

# --- Configuration ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("CORE_RL_TASK", "zombie_hunter")
BENCHMARK = "core_rl"
MAX_STEPS = 8

# --- Logging Helpers (MANDATORY FORMAT) ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    done_str = str(done).lower()
    err_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rew_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rew_str}", flush=True)

# --- LLM Logic ---
def get_agent_action(client: OpenAI, observation: dict) -> Action:
    """Ask the LLM to decide on a FinOps action based on the dashboard."""
    system_prompt = textwrap.dedent("""
        You are a Cloud FinOps Engineer. Your goal is to reduce cloud costs.
        You can use: stop, resize, or no_op.
        CRITICAL: Never 'stop' a resource marked as is_critical=True.
        Respond ONLY with a JSON object: {"command": "...", "resource_id": "...", "new_tier": "..."}
    """).strip()

    user_prompt = f"Current State: {json.dumps(observation, indent=2)}\nWhat is your next move?"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return Action(**data)
    except Exception as e:
        # Fallback to no_op on error
        return Action(command="no_op", resource_id="none")

async def main():
    # 1. Setup Clients
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    # EnvClient connects to your local FastAPI server or remote HF Space
    remote_url = os.getenv("PING_URL", "http://localhost:7860")
    env = EnvClient(remote_url)

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0
    
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # 2. Start Episode
        obs = await env.reset(task_id=TASK_NAME)
        
        for step in range(1, MAX_STEPS + 1):
            # 3. Get LLM decision
            action_obj = get_agent_action(client, obs.model_dump())
            
            # 4. Step Environment
            result = await env.step(action_obj)
            
            # 5. Log Step
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            log_step(step, action_obj.command, reward, result.done, None)
            
            obs = result.observation
            if result.done:
                break
        
        # Calculate final score (normalized 0-1)
        final_reward = sum(rewards)
        # We consider it a success if we didn't fail critically and saved some money
        success = final_reward > 0
        score = min(max(final_reward / 2.0, 0.0), 1.0) 

    finally:
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())