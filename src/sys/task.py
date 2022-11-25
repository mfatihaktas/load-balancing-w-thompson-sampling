class Task:
    def __init__(
        self,
        _id: str,
        serv_time: float,
    ):
        self._id = _id
        self.serv_time = serv_time

        self.node_id = None

    def __repr__(self):
        # return (
        #     "Msg( \n"
        #     f"\t id= {self._id} \n"
        #     ")"
        # )

        return f"Task(id= {self._id}, serv_time= {self.serv_time})"
