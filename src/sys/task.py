class Task:
    def __init__(
        self,
        _id: str,
        service_time: float,
    ):
        self._id = _id
        self.service_time = service_time

        self.node_id = None

    def __repr__(self):
        # return (
        #     "Msg( \n"
        #     f"\t id= {self._id} \n"
        #     ")"
        # )

        return f"Task(id= {self._id}, service_time= {self.service_time})"
