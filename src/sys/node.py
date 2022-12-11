import simpy

from src.utils.debug import *


class Node:
    def __init__(self, env: simpy.Environment, _id: str):
        self.env = env
        self._id = _id

    def __repr__(self):
        return "Node( \n" f"\t id= {self._id} \n" ")"

    @abc.abstractmethod
    def work_left(self):
        return
