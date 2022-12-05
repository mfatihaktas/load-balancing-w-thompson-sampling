import pytest
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
from src.sim import sim

from src.utils.debug import *


@pytest.fixture(scope="module")
def env() -> simpy.Environment:
    return simpy.Environment()


@pytest.fixture(scope="module")
def num_servers() -> int:
    return 2


def test_ThompsonSampling_slidingWin(
    env: simpy.Environment,
    num_servers: int,
):
    server_list = sim.get_servers(env=env, num_servers=num_servers)

    sching_agent = ts_module.ThompsonSampling_slidingWin(
        node_id_list=[s._id for s in server_list],
        win_len=100,
    )

    sim_result = sim.sim(
        env=env,
        server_list=server_list,
        sching_agent=sching_agent,
        inter_task_gen_time_rv=random_variable.Exponential(mu=1),
        task_service_time_rv=random_variable.DiscreteUniform(min_value=1, max_value=1),
        num_tasks_to_recv=100,
    )
    log(INFO, "", sim_result=sim_result)
