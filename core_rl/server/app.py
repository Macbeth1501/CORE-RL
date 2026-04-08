# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the CORE-RL FinOps Environment.
"""

import os
try:
    # Meta uses 'create_app' to wrap the environment in a web interface
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install with 'uv sync' or 'pip install openenv-core'"
    ) from e

# Import your custom FinOps logic
try:
    # This matches the folder structure we've built
    from .models import Action, Observation
    from .core_rl_environment import CoreRLEnvironment
except (ModuleNotFoundError, ImportError):
    # Fallback for different execution contexts
    from models import Action, Observation
    from server.core_rl_environment import CoreRLEnvironment

# Create the app
app = create_app(
    CoreRLEnvironment,
    Action,
    Observation,
    env_name="core_rl",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for running the server."""
    import uvicorn
    # Note: We use port 7860 for Hugging Face compatibility
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Default to 7860 for HF Spaces
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)