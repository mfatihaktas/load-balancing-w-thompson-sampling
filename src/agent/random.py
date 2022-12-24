import random

from src.agent import agent
from src.sys import node

from src.utils.debug import *


class AssignToRandom(agent.SchingAgent):
    def __init__(self, node_list: list[node.Node]):
        self.node_list = node_list

    def __repr__(self):
        return (
            "AssignToRandom( \n"
            f"\t num_nodes= {len(self.node_list)} \n"
            ")"
        )

    def node_id_to_assign(self) -> str:
        return random.sample(self.node_list, 1)[0]._id
