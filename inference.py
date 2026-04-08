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
    remote_url = os.getenv("PING_URL", "http://localhost:7860")

    class FinOpsClient(EnvClient):
        def _parse_result(self, data):
            class Result:
                def __init__(self, d):
                    # ROBUST EXTRACTION:
                    # If 'd' is the whole result, it has an 'observation' key.
                    # We need to extract that inner dictionary to build the Observation object.
                    if isinstance(d, dict) and "observation" in d:
                        obs_payload = d["observation"]
                    else:
                        obs_payload = d
                    
                    # Sometimes the framework nests it twice, let's be safe:
                    if isinstance(obs_payload, dict) and "observation" in obs_payload:
                        obs_payload = obs_payload["observation"]

                    # Now initialize our Pydantic model with the raw fields (resources, etc.)
                    self.observation = Observation(**obs_payload)
                    
                    # Extract reward and done from the outer dictionary
                    self.reward = d.get("reward", 0.0) if isinstance(d, dict) else 0.0
                    self.done = d.get("done", False) if isinstance(d, dict) else False
                    self.info = d.get("info", {}) if isinstance(d, dict) else {}
            return Result(data)

        def _parse_state(self, data): 
            return data
            
        def _step_payload(self, action): 
            return action.model_dump()

    env = FinOpsClient(remote_url)

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0
    
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # 2. Start Episode
        # Fixed: env.reset returns the Result object, we need the observation inside it
        reset_result = await env.reset(task_id=TASK_NAME)
        obs = reset_result.observation 
        
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
            
            # Update obs for the next iteration
            obs = result.observation
            if result.done:
                break
        
        # Calculate final score (normalized 0-1)
        final_reward = sum(rewards)
        
        # Max reward for zombie_hunter is 0.2 * 3 = 0.6
        # Max reward for fleet_resizer is 0.1 * 5 = 0.5
        # We normalize against these targets to get a score closer to 1.0
        if TASK_NAME == "zombie_hunter":
            score = final_reward / 0.6
        elif TASK_NAME == "fleet_resizer":
            score = final_reward / 0.5
        else:
            score = final_reward / 1.0 # fallback for hard task

        score = min(max(score, 0.0), 1.0) # Clamp between 0 and 1
        success = score >= 0.8 # Consider it a success if we got most of them

    finally:
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())