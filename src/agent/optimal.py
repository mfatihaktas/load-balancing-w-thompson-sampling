from src.agent import agent
from src.prob import random_variable
from src.sys import node

from src.utils.debug import *


class AssignToLeastWorkLeft(agent.SchingAgent):
    def __init__(self, node_list: list[node.Node]):
        self.node_list = node_list

    def __repr__(self):
        return (
            "AssignToLeastWorkLeft( \n"
            f"\t num_nodes= {len(self.node_list)} \n"
            ")"
        )

    def node_id_to_assign(self) -> str:
        node_w_least_work = None
        least_work = float("Inf")
        for node in self.node_list:
            work_left = node.work_left()

            if work_left < least_work:
                least_work = work_left
                node_w_least_work = node

        return node_w_least_work._id


class AssignToNoisyLeastWorkLeft(AssignToLeastWorkLeft):
    def __init__(self, node_list: list[node.Node], noise_rv: random_variable.RandomVariable):
        super().__init__(node_list=node_list)
        self.noise_rv = noise_rv

        check(self.noise_rv.min_value > 0 and self.noise_rv.max_value > 0,
              f"Improper noise_rv= {noise_rv}")

    def __repr__(self):
        return (
            "AssignToNoisyLeastWorkLeft( \n"
            f"\t num_nodes= {len(self.node_list)} \n"
            f"\t noise_rv= {self.noise_rv} \n"
            ")"
        )

    def node_id_to_assign(self) -> str:
        node_w_least_work = None
        least_work = float("Inf")
        for node in self.node_list:
            noise = self.noise_rv.sample()
            log(DEBUG, f"Sampled noise= {noise}", node=node)
            work_left = node.work_left() * noise

            if work_left < least_work:
                least_work = work_left
                node_w_least_work = node

        return node_w_least_work._id


class AssignToFewestTasksLeft(agent.SchingAgent):
    def __init__(self, node_list: list[node.Node]):
        self.node_list = node_list

    def __repr__(self):
        return (
            "AssignToFewestTasksLeft( \n"
            f"\t num_nodes= {len(self.node_list)} \n"
            ")"
        )

    def node_id_to_assign(self) -> str:
        node_w_fewest_tasks_left = None
        fewest_tasks_left = float("Inf")
        for node in self.node_list:
            num_tasks_left = node.num_tasks_left()

            if num_tasks_left < fewest_tasks_left:
                fewest_tasks_left = num_tasks_left
                node_w_fewest_tasks_left = node

        return node_w_fewest_tasks_left._id
