from models import BaseModel


class NLL:
    def __init__(self, model: BaseModel, data_loc=None):
        self.model = model
        self.data_loc = data_loc

    def get_score(self):
        self.model.get_nll(aaaaa, self.data_loc)
