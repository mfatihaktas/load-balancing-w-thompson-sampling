import abc
import simpy

from src.utils.debug import *


class Node:
    def __init__(self, env: simpy.Environment, _id: str):
        self.env = env
        self._id = _id

    def __repr__(self):
        return f"Node(id= {self._id})"

    @abc.abstractmethod
    def num_tasks_left(self):
        return

    @abc.abstractmethod
    def work_left(self):
        return
