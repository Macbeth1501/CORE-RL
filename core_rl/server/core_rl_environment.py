import uuid
from typing import Dict, List, Optional
from openenv.core.env_server import Environment
from .models import Action, Observation, Resource, Reward

class CoreRLEnvironment(Environment):
    def __init__(self, id: str = "core_rl", **kwargs):
        super().__init__() 
        self.resources: Dict[str, Resource] = {}
        self.budget_limit = 500.0
        self.steps_taken = 0
        self.max_steps = 8
        self.is_done = False
        self.last_msg = ""

    def reset(self, task_id: str = "zombie_hunter") -> Observation:
        """Initializes a specific FinOps scenario."""
        self.resources = {}
        self.steps_taken = 0
        self.is_done = False
        self.last_msg = f"Task Started: {task_id}"

        # Scenario setup logic
        if task_id == "zombie_hunter":
            # EASY: Terminal 3 'Zombie' VMs (0% usage)
            for i in range(1, 4):
                self._add_res(f"zombie-vm-{i}", "VM", 0.0, 45.0, False)
            self._add_res("prod-db", "Database", 0.85, 300.0, True)

        elif task_id == "fleet_resizer":
            # MEDIUM: 5 Oversized VMs (5% usage)
            for i in range(1, 6):
                self._add_res(f"oversized-vm-{i}", "VM", 0.05, 120.0, False)
            self._add_res("main-web", "VM", 0.4, 80.0, True)

        else:  # budget_breach (HARD)
            # 10 mixed resources, high spend, strict budget
            for i in range(1, 11):
                self._add_res(f"app-node-{i}", "VM", 0.1, 100.0, False)
            self._add_res("critical-gateway", "VM", 0.9, 200.0, True)

        return self._get_obs(reward=0.0)

    def step(self, action: Action) -> (Observation, float, bool, dict):
        """Processes FinOps actions and calculates rewards."""
        self.steps_taken += 1
        step_reward = 0.0
        
        # 1. Action Execution
        res_id = action.resource_id
        if action.command == "stop":
            if res_id in self.resources:
                target = self.resources[res_id]
                if target.is_critical:
                    # SAFETY PENALTY: Killing critical infra ends the episode
                    step_reward = -1.0
                    self.is_done = True
                    self.last_msg = f"CRITICAL FAILURE: Stopped {res_id}!"
                else:
                    savings = target.cost_per_hour
                    del self.resources[res_id]
                    # EFFICIENCY REWARD: Give 0.2 for stopping waste
                    step_reward = 0.2 
                    self.last_msg = f"Stopped {res_id}. Saved ${savings}/hr."
            else:
                self.last_msg = "Error: Resource ID not found."

        elif action.command == "resize":
            if res_id in self.resources:
                old_cost = self.resources[res_id].cost_per_hour
                # Simplified resize: tier change cuts cost by 50%
                self.resources[res_id].cost_per_hour *= 0.5
                step_reward = 0.1
                self.last_msg = f"Resized {res_id} to {action.new_tier}. Cost halved."
            else:
                self.last_msg = "Error: Resource ID not found."
        
        elif action.command == "no_op":
            self.last_msg = "Agent skipped a turn."

        # 2. Episode Boundary Check
        if self.steps_taken >= self.max_steps:
            self.is_done = True

        return self._get_obs(reward=step_reward), step_reward, self.is_done, {"reason": self.last_msg}

    def _add_res(self, id, rtype, util, cost, critical):
        self.resources[id] = Resource(
            id=id, type=rtype, utilization=util, 
            cost_per_hour=cost, is_critical=critical
        )

    def _get_obs(self, reward: float = 0.0) -> Observation:
        current_spend = sum(r.cost_per_hour for r in self.resources.values())
        health = 1.0 if any(r.is_critical for r in self.resources.values()) else 0.0
        return Observation(
            resources=list(self.resources.values()),
            current_hourly_spend=current_spend,
            budget_limit=self.budget_limit,
            system_health=health,
            last_action_status=self.last_msg,
            reward=reward,
            done=self.is_done  # PASS THE CURRENT STATE HERE
        )
    def state(self) -> dict:
        """Returns the internal state for logging/debugging."""
        return {
            "resources_count": len(self.resources),
            "steps_taken": self.steps_taken,
            "budget_limit": self.budget_limit,
            "is_done": self.is_done,
            "last_msg": self.last_msg
        }