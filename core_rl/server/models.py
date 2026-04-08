from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Resource(BaseModel):
    id: str = Field(..., description="Unique ID: srv-01, db-02, etc.")
    type: Literal["VM", "Database", "Storage"]
    utilization: float = Field(..., description="0.0 to 1.0 (CPU/Memory usage)")
    cost_per_hour: float = Field(..., description="Hourly cost in USD")
    is_critical: bool = Field(..., description="If True, stopping this causes failure")

class Action(BaseModel):
    command: Literal["stop", "resize", "no_op"]
    resource_id: str
    new_tier: Optional[Literal["micro", "small", "large"]] = None

class Observation(BaseModel):
    resources: List[Resource]
    current_hourly_spend: float
    budget_limit: float
    system_health: float
    last_action_status: str
    reward: float = 0.0
    done: bool = False

class Reward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    reason: str

class EnvStepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}