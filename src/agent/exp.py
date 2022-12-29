import dataclasses

from src.sys import task as task_module


@dataclasses.dataclass
class Exp:
    service_time: float
    wait_time: float


def get_exp(time_epoch: float, task: task_module.Task) -> Exp:
    return Exp(
        service_time=task.service_time,
        wait_time=time_epoch - task.arrival_time - task.service_time,
    )
