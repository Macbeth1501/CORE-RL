import asyncio
import os
import textwrap
import json
from typing import List, Optional
from openai import OpenAI
from openenv.core.env_client import EnvClient
from server.models import Action, Observation

# --- CONFIGURATION (STRICT CHECKLIST COMPLIANCE) ---
# Defaults are allowed for these two
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# NO DEFAULT allowed for HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")

# Internal helper for your Space URL
REMOTE_ENV_URL = os.getenv("PING_URL", "https://macbeth1501-core-rl.hf.space")

BENCHMARK = "core_rl"
MAX_STEPS = 8

# --- LOGGING HELPERS (STRICT START/STEP/END FORMAT) ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    done_str = str(done).lower()
    err_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rew_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rew_str}", flush=True)

# --- LLM LOGIC ---
def get_agent_action(client: OpenAI, observation: dict) -> Action:
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
    except:
        return Action(command="no_op", resource_id="none")

async def main():
    # ALL LLM calls use the OpenAI client configured via these variables
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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

    env = FinOpsClient(REMOTE_ENV_URL)
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
            
            final_reward = sum(rewards)
            if current_task == "zombie_hunter":
                score = final_reward / 0.6
            elif current_task == "fleet_resizer":
                score = final_reward / 0.5
            else:
                score = final_reward / 1.0 

            score = min(max(score, 0.001), 0.999)
            success = score >= 0.7 

        except Exception as e:
            pass
        finally:
            log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())