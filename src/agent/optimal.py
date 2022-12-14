from src.agent import agent
from src.sys import node

class AssignToLeastWorkLeft(agent.SchingAgent):
    def __init__(self, node_list: list[node.Node]):
        self.node_list = node_list

    def __repr__(self):
        return (
            f"AssignToLeastWorkLeft( \n"
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
