import os
from openenv.core.env_server.http_server import create_app

# Import your custom FinOps logic
try:
    from .models import Action, Observation
    from .core_rl_environment import CoreRLEnvironment
except (ModuleNotFoundError, ImportError):
    from models import Action, Observation
    from server.core_rl_environment import CoreRLEnvironment

# This is the "Bulletproof" way: 
# We define a function that returns your environment instance.
def env_factory(**kwargs):
    try:
        return CoreRLEnvironment()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

app = create_app(
    env_factory,  # Pass the function directly
    Action,
    Observation,
    env_name="core_rl",
    max_concurrent_envs=1,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)