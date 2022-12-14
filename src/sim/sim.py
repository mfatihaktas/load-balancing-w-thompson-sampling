import dataclasses
import numpy
import simpy

from typing import Callable

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


@dataclasses.dataclass
class SimResult:
    ET: float
    std_T: float


def sim(
    env: simpy.Environment,
    num_servers: int,
    inter_task_gen_time_rv: random_variable.RandomVariable,
    task_service_time_rv: random_variable.RandomVariable,
    num_tasks_to_recv: int,
    sching_agent_given_server_list: Callable[[list[server_module.Server]], agent_module.SchingAgent],
    num_sim_runs: int = 1,
) -> SimResult:
    log(DEBUG, "Started",
        num_servers=num_servers,
        inter_task_gen_time_rv=inter_task_gen_time_rv,
        task_service_time_rv=task_service_time_rv,
        num_tasks_to_recv=num_tasks_to_recv,
        sching_agent_given_server_list=sching_agent_given_server_list,
        num_sim_runs=num_sim_runs,
    )

    def sim_run_once() -> list[float]:
        sink = sink_module.Sink(env=env, _id="sink")

        server_list = [
            server_module.Server(env=env, _id=f"s{i}", sink=sink) for i in range(num_servers)
        ]

        sching_agent = sching_agent_given_server_list(server_list=server_list)

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

        return sink.task_response_time_list

    task_response_time_list = []
    for i in range(num_sim_runs):
        log(INFO, f">> sim-{i}")

        task_response_time_list_ = sim_run_once()
        ET = numpy.mean(task_response_time_list_)
        std_T = numpy.std(task_response_time_list_)
        log(INFO, "", ET=ET, std_T=std_T)

        task_response_time_list.extend(task_response_time_list_)

    ET = numpy.mean(task_response_time_list)
    std_T = numpy.std(task_response_time_list)
    return SimResult(ET=ET, std_T=std_T)
