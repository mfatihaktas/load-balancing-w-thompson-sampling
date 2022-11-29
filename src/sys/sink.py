import simpy

from src.agent import agent
from src.sys import (
    node,
    task as task_module,
)
from src.utils.debug import *


class Sink(node.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        sching_agent: agent.SchingAgent = None,
        num_tasks_to_recv: int = None,
    ):
        super().__init__(env=env, _id=_id)
        self.sching_agent = sching_agent
        self.num_tasks_to_recv = num_tasks_to_recv

        self.task_store = simpy.Store(env)
        self.recv_tasks_proc = env.process(self.recv_tasks())

    def __repr__(self):
        return f"Sink(id= {self._id})"

    def put(self, task: task_module.Task):
        slog(DEBUG, self.env, self, "recved", task=task)

        self.task_store.put(task)

    def recv_tasks(self):
        slog(DEBUG, self.env, self, "started")

        num_tasks_recved = 0
        while True:
            task = yield self.task_store.get()
            num_tasks_recved += 1
            slog(DEBUG, self.env, self, "recved", task=task, num_tasks_recved=num_tasks_recved)

            if self.sching_agent:
                response_time = self.env.now - task.arrival_time
                self.sching_agent.record_cost(node_id=task.node_id, cost=response_time)

            if num_tasks_recved >= self.num_tasks_to_recv:
                slog(DEBUG, self.env, self, "recved requested # tasks", num_tasks_recved=num_tasks_recved)
                break

        slog(DEBUG, self.env, self, "done")
