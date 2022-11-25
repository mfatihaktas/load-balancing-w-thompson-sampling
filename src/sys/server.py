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
        sink: node.Node,
    ):
        super().__init__(env=env, _id=_id)
        self.sink = sink

        self.task_store = simpy.Store(env)
        self.recv_tasks_proc = env.process(self.recv_tasks())

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

        num_tasks_proced = 0
        while True:
            task = yield self.task_store.get()
            yield self.env.timeout(task.serv_time)

            num_tasks_proced += 1
            slog(DEBUG, self.env, self,
                "processed",
                task=task,
                num_tasks_proced=num_tasks_proced,
                queue_len=len(self.task_store),
            )

        slog(DEBUG, self.env, self, "done")
