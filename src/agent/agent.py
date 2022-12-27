import abc

from src.sys import node


class SchingAgent(abc.ABC):
    def __init__(self):
        pass

    def __repr__(self):
        return "SchingAgent()"

    @abc.abstractmethod
    def node_id_to_assign(self, time_epoch: float=None):
        pass


class SchingAgent_wOnlineLearning(SchingAgent):
    def __init__(self, node_list: list[node.Node]):
        self.node_list = node_list

        self.node_id_list = [node._id for node in self.node_list]

    def __repr__(self):
        return (
            "SchingAgent_wOnlineLearning( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            ")"
        )

    @abc.abstractmethod
    def record_cost(self, node_id: str, cost: float):
        return None
