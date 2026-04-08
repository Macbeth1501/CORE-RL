from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# The individual Cloud Resource model
class Resource(BaseModel):
    id: str = Field(..., description="Unique ID: srv-01, db-02, etc.")
    type: Literal["VM", "Database", "Storage"]
    utilization: float = Field(..., description="0.0 to 1.0 (CPU/Memory usage)")
    cost_per_hour: float = Field(..., description="Hourly cost in USD")
    is_critical: bool = Field(..., description="If True, stopping this causes failure")

# THE ACTION: This is what the Agent sends to the env
class Action(BaseModel):
    command: Literal["stop", "resize", "no_op"]
    resource_id: str
    new_tier: Optional[Literal["micro", "small", "large"]] = None

# THE OBSERVATION: This is what the Agent receives from the env
class Observation(BaseModel):
    resources: List[Resource]
    current_hourly_spend: float
    budget_limit: float
    system_health: float = Field(..., description="Overall health (0.0 to 1.0)")
    last_action_status: str

# THE REWARD: Required by OpenEnv spec
class Reward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    reason: str