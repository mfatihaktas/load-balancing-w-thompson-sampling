import pytest
import simpy

from src.agent import agent as agent_module
from src.prob import random_variable
from src.sim import sim as sim_module


@pytest.fixture(scope="module")
def env() -> simpy.Environment:
    return simpy.Environment()


@pytest.fixture(scope="module")
def num_servers() -> int:
    return 2


@pytest.fixture(scope="module")
def num_servers() -> int:
    return 2


@pytest.fixture(scope="module")
def inter_task_gen_time_rv() -> random_variable.RandomVariable:
    return random_variable.Exponential(mu=1)


@pytest.fixture(scope="module")
def task_service_time_rv() -> random_variable.RandomVariable:
    return random_variable.DiscreteUniform(min_value=1, max_value=1)
