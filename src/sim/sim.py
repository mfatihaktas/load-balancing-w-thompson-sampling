import simpy

from src.sys import (
    scheduler as scheduler_module,
    server as server_module,
    sink as sink_module,
    source as source_module,
)
from src.agent import (
    agent as agent_module,
    ts as ts_module,
)
from src.prob import random_variable

from src.utils.debug import *


def sim(
    env: simpy.Environment,
    server_list: list[server_module.Server],
    sink: sink_module.Sink,
    sching_agent: agent_module.SchingAgent,
    inter_task_gen_time_rv: random_variable.RandomVariable,
    task_service_time_rv: random_variable.RandomVariable,
    num_tasks_to_recv: int,
):
    scher = scheduler_module.Scheduler(
        env=env,
        _id="scher",
        node_list=server_list,
        sching_agent=sching_agent,
    )

    source = source_module.Source(
        env=env,
        _id="source",
        inter_task_gen_time_rv=inter_task_gen_time_rv,
        task_service_time_rv=task_service_time_rv,
        next_hop=scher,
    )

    sink.sching_agent = sching_agent
    sink.num_tasks_to_recv = num_tasks_to_recv

    env.run(until=sink.recv_tasks_proc)
