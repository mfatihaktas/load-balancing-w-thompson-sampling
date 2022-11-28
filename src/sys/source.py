import simpy

from src.utils.debug import *
from src.prob import random_variable
from src.sys import (
    node,
    task as task_module,
)


class Source(node.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        inter_msg_gen_time_rv: random_variable.RandomVariable,
        task_service_time_rv: random_variable.RandomVariable,
        next_hop: node.Node,
        num_msgs_to_send: int = None,
    ):
        super().__init__(env=env, _id=_id)
        self.inter_msg_gen_time_rv = inter_msg_gen_time_rv
        self.task_service_time_rv = task_service_time_rv
        self.next_hop = next_hop
        self.num_msgs_to_send = num_msgs_to_send

        self.send_messages_proc = env.process(self.send_tasks())

    def __repr__(self):
        # return (
        #     "Source( \n"
        #     f"{super().__repr__()} \n"
        #     f"\t dst_id= {self.dst_id} \n"
        #     f"\t num_msgs_to_send= {self.num_msgs_to_send} \n"
        #     f"\t inter_msg_gen_time_rv= {self.inter_msg_gen_time_rv} \n"
        #     ")"
        # )

        return f"Source(id= {self._id})"

    def send_tasks(self):
        slog(DEBUG, self.env, self, "started")

        task_id = 0
        while True:
            inter_msg_gen_time = self.inter_msg_gen_time_rv.sample()
            slog(DEBUG, self.env, self, "waiting",
                 inter_msg_gen_time=inter_msg_gen_time
            )
            yield self.env.timeout(inter_msg_gen_time)

            task = task_module.Task(_id=task_id, serv_time=self.task_service_time_rv.sample())

            slog(DEBUG, self.env, self, "sending", task=task)
            self.next_hop.put(task)

            task_id += 1
            if self.num_msgs_to_send and task_id >= self.num_msgs_to_send:
                break

        slog(DEBUG, self.env, self, "started")
