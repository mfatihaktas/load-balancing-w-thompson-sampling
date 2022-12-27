import simpy

from src.utils.debug import *
from src.sys import (
    node,
    task as task_module,
)


class Server(node.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        sink: node.Node = None,
    ):
        super().__init__(env=env, _id=_id)
        self.sink = sink

        self.task_in_serv = None
        self.serv_start_time = None
        self.task_store = simpy.Store(env)
        self.recv_tasks_proc = env.process(self.recv_tasks())

    def __repr__(self):
        # return (
        #     "Server( \n"
        #     f"{super().__repr__()} \n"
        #     ")"
        # )

        return f"Server(id= {self._id})"

    def repr_w_state(self):
        return (
            "Server( \n"
            f"\t num_tasks_left= {self.num_tasks_left()} \n"
            f"\t work_left= {self.work_left()} \n"
            ")"
        )

    def num_tasks_left(self) -> int:
        return len(self.task_store.items) + int(self.task_in_serv is not None)

    def work_left(self) -> float:
        remaining_serv_time = 0
        if self.task_in_serv:
            remaining_serv_time = self.task_in_serv.service_time - (self.env.now - self.serv_start_time)

        return remaining_serv_time + sum(task.service_time for task in self.task_store.items)

    def put(self, task: task_module.Task):
        slog(DEBUG, self.env, self, "recved", task=task)

        task.node_id = self._id
        self.task_store.put(task)

    def recv_tasks(self):
        slog(DEBUG, self.env, self, "started")

        num_tasks_proced = 0
        while True:
            self.task_in_serv = yield self.task_store.get()
            self.serv_start_time = self.env.now
            yield self.env.timeout(self.task_in_serv.service_time)

            num_tasks_proced += 1
            slog(DEBUG, self.env, self,
                "processed",
                task_in_serv=self.task_in_serv,
                num_tasks_proced=num_tasks_proced,
                queue_len=len(self.task_store.items),
            )

            self.sink.put(self.task_in_serv)
            self.task_in_serv = None

        slog(DEBUG, self.env, self, "done")
