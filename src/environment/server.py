import simpy

from src.debug_utils import *
from src.sim import (
    node,
    task as task_module,
)


class Server(node.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
    ):
        super().__init__(env=env, _id=_id)

        self.task_store = simpy.Store(env)
        self.process_recv_tasks = env.process(self.recv_tasks())

    def __repr__(self):
        # return (
        #     "Server( \n"
        #     f"{super().__repr__()} \n"
        #     ")"
        # )

        return f"Server(id= {self._id})"

    def put(self, task: task_module.Task):
        slog(DEBUG, self.env, self, "recved", task=task)

        self.task_store.put(task)

    def recv_tasks(self):
        slog(DEBUG, self.env, self, "started")

        num_tasks_recved = 0
        while True:
            task = yield self.task_store.get()
            num_tasks_recved += 1
            slog(
                DEBUG,
                self.env,
                self,
                "processed",
                num_tasks_recved=num_tasks_recved,
                task=task,
            )

        slog(DEBUG, self.env, self, "done")
