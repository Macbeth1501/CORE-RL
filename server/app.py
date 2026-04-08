import os
import uvicorn
from openenv.core.env_server.http_server import create_app
from fastapi.responses import RedirectResponse

# Import your custom FinOps logic
try:
    # Works when imported as a package
    from .models import Action, Observation
    from .core_rl_environment import CoreRLEnvironment
except (ModuleNotFoundError, ImportError):
    # Works when run as a script from the root
    from server.models import Action, Observation
    from server.core_rl_environment import CoreRLEnvironment

def env_factory(**kwargs):
    try:
        return CoreRLEnvironment()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

# The app object must be at the module level for Uvicorn/Grader to find
app = create_app(
    env_factory,
    Action,
    Observation,
    env_name="core_rl",
    max_concurrent_envs=1,
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

def main():
    """
    Standard entry point required by the grader for multi-mode deployment.
    """
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()