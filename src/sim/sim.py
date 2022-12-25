import dataclasses
import joblib
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


@dataclasses.dataclass(repr=False)
class SimResult:
    t_l: list[float]

    ET: float = None
    std_T: float = None

    def __repr__(self):
        return (
            "SimResult( \n"
            f"\t ET= {self.ET} \n"
            f"\t std_T= {self.std_T} \n"
            ")"
        )

    def __post_init__(self):
        self.ET = numpy.mean(self.t_l)
        self.std_T = numpy.std(self.t_l)


def combine_sim_results(sim_result_list: list[SimResult]) -> SimResult:
    t_l = []
    for sim_result in sim_result_list:
        t_l.extend(sim_result.t_l)

    return SimResult(t_l=t_l)


def sim(
    env: simpy.Environment,
    num_servers: int,
    inter_task_gen_time_rv: random_variable.RandomVariable,
    task_service_time_rv: random_variable.RandomVariable,
    num_tasks_to_recv: int,
    sching_agent_given_server_list: Callable[[list[server_module.Server]], agent_module.SchingAgent],
    sim_result_list: list[SimResult],
):
    log(DEBUG, "Started",
        num_servers=num_servers,
        inter_task_gen_time_rv=inter_task_gen_time_rv,
        task_service_time_rv=task_service_time_rv,
        num_tasks_to_recv=num_tasks_to_recv,
    )

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

    sim_result = SimResult(t_l=sink.task_response_time_list)
    log(INFO, "Done", sim_result=sim_result)

    sim_result_list.append(sim_result)


def sim_w_joblib(
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

    sim_result_list = []
    if num_sim_runs == 1:
        sim(
            env=env,
            num_servers=num_servers,
            inter_task_gen_time_rv=inter_task_gen_time_rv,
            task_service_time_rv=task_service_time_rv,
            num_tasks_to_recv=num_tasks_to_recv,
            sching_agent_given_server_list=sching_agent_given_server_list,
            sim_result_list=sim_result_list,
        )

    else:
        joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(sim)(
                env=env,
                num_servers=num_servers,
                inter_task_gen_time_rv=inter_task_gen_time_rv,
                task_service_time_rv=task_service_time_rv,
                num_tasks_to_recv=num_tasks_to_recv,
                sching_agent_given_server_list=sching_agent_given_server_list,
                sim_result_list=sim_result_list,
            )
            for i in range(num_sim_runs)
        )

    sim_result = combine_sim_results(sim_result_list=sim_result_list)
    log(INFO, "Done", sim_result=sim_result)
    return sim_result
