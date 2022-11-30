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

from src.utils.debug import *


@pytest.fixture(scope="module")
def env() -> simpy.Environment:
    return simpy.Environment()


@pytest.fixture(scope="module")
def num_servers() -> int:
    return 2


@pytest.fixture(scope="module")
def sink(env: simpy.Environment) -> sink_module.Sink:
    return sink_module.Sink(env=env, _id="sink")


@pytest.fixture(scope="module")
def server_list(
    env: simpy.Environment,
    num_servers: int,
    sink: sink_module.Sink,
) -> list[server_module.Server]:
    return [
        server_module.Server(env=env, _id=f"s{i}", sink=sink) for i in range(num_servers)
    ]


def test_scheduler_ts_sliding_win(
    env: simpy.Environment,
    server_list: list[server_module.Server],
    sink: sink_module.Sink,
) -> list[server_module.Server]:
    sching_agent = ts_module.ThompsonSampling_slidingWin(
        node_id_list=[s._id for s in server_list],
        win_len=100,
    )

    scher = scheduler_module.Scheduler(
        env=env,
        _id="scher",
        node_list=server_list,
        sching_agent=sching_agent,
    )

    source = source_module.Source(
        env=env,
        _id="source",
        inter_msg_gen_time_rv=random_variable.Exponential(mu=1),
        task_service_time_rv=random_variable.DiscreteUniform(min_value=1, max_value=1),
        next_hop=scher,
    )

    sink.sching_agent = sching_agent
    sink.num_tasks_to_recv = 100

    env.run(until=sink.recv_tasks_proc)
