import simpy

from src.utils.debug import *
from src.agent import agent
from src.sys import (
    node as node_module,
    task as task_module,
)


class Scheduler(node_module.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        node_list: list[node_module.Node],
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

    def schedule(self, task: task_module.Task):
        slog(DEBUG, self.env, self, "started")

        node_id = self.sching_agent.node_id_to_assign(time_epoch=self.env.now)

        slog(DEBUG, self.env, self, "will schedule task",
             num_tasks_sched=self.num_tasks_sched,
             task=task,
             node_id=node_id,
        )
        self.id_to_node_map[node_id].put(task)
        self.num_tasks_sched += 1

        slog(DEBUG, self.env, self, "done")
