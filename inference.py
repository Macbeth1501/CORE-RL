import asyncio
import os
import textwrap
import json
from typing import List, Optional
from openai import OpenAI
from openenv.core.env_client import EnvClient
# Change this line (approx line 13)
from core_rl.server.models import Action, Observation  # Added Observation and fixed path

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
    #remote_url = os.getenv("PING_URL", "http://localhost:7860")
    remote_url = os.getenv("PING_URL", "https://macbeth1501-core-rl.hf.space")

    class FinOpsClient(EnvClient):
        def _parse_result(self, data):
            class Result:
                def __init__(self, d):
                    obs_payload = d.get("observation", d)
                    if isinstance(obs_payload, dict) and "observation" in obs_payload:
                        obs_payload = obs_payload["observation"]
                    self.observation = Observation(**obs_payload)
                    self.reward = d.get("reward", 0.0)
                    self.done = d.get("done", False)
                    self.info = d.get("info", {})
            return Result(data)
        def _parse_state(self, data): return data
        def _step_payload(self, action): return action.model_dump()

    env = FinOpsClient(remote_url)
    
    # LIST OF TASKS TO EVALUATE
    tasks_to_run = ["zombie_hunter", "fleet_resizer", "budget_breach"]

    for current_task in tasks_to_run:
        rewards = []
        steps_taken = 0
        success = False
        score = 0.0
        
        log_start(current_task, BENCHMARK, MODEL_NAME)

        try:
            reset_result = await env.reset(task_id=current_task)
            obs = reset_result.observation 
            
            for step in range(1, MAX_STEPS + 1):
                action_obj = get_agent_action(client, obs.model_dump())
                result = await env.step(action_obj)
                
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                log_step(step, action_obj.command, reward, result.done, None)
                
                obs = result.observation
                if result.done:
                    break
            
            # --- TASK-SPECIFIC SCORING ---
            final_reward = sum(rewards)
            if current_task == "zombie_hunter":
                score = final_reward / 0.6  # 3 zombies * 0.2
            elif current_task == "fleet_resizer":
                score = final_reward / 0.5  # 5 resizes * 0.1
            else: # budget_breach
                # In budget breach, the agent needs to save ~$1000. 
                # Let's say saving 5 nodes ($1.0 reward) is a good baseline.
                score = final_reward / 1.0 

            score = min(max(score, 0.0), 1.0) 
            success = score >= 0.7 

        except Exception as e:
            print(f"[DEBUG] Task {current_task} failed: {e}")
        finally:
            log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())