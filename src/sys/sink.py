import simpy

from src.debug_utils import *
from src.sim import (
    node,
    task as task_module,
)


class Sink(node.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
    ):
        super().__init__(env=env, _id=_id)

        self.task_store = simpy.Store(env)
        # TODO:
        self.sching_agent = None

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

            self.sching_agent.inform(response_time_sample=task.response_time)

        slog(DEBUG, self.env, self, "done")
