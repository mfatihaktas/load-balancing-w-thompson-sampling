import simpy

from src.utils.debug import *
from src.sim import (
    node,
    task as task_module,
)


class Scheduler(node.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        node_list: list[node.Node],
        sching_agent: agent.SchingAgent,
    ):
        super().__init__(env=env, _id=_id)
        self.sching_agent = sching_agent
        self.id_to_node_map = {node._id: node for node in node_list}

        self.num_tasks_sched = 0

    def __repr__(self):
        # return (
        #     "Scheduler( \n"
        #     f"{super().__repr__()} \n"
        #     ")"
        # )

        return f"Scheduler(id= {self._id})"

    def put(self, task: task_module.Task):
        slog(DEBUG, self.env, self, "recved", task=task)

        self.schedule(task)
        self.task_store.put(task)

    def schedule(self, task: task_module.Task):
        slog(DEBUG, self.env, self, "started")

        node_id = self.sching_algo.node_to_schedule(task=task)

        self.id_to_node_map[node_id].put(task)
        slog(DEBUG, self.env, self, "scheduled task",
             num_tasks_sched=self.num_tasks_sched,
             task=task,
        )
        self.num_tasks_sched += 1

        slog(DEBUG, self.env, self, "done")