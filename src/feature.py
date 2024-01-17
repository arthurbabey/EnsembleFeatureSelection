class Feature:
    def __init__(self, name, score=None, selected=False):
        self.name = name
        self.score = score
        self.selected = selected

    def set_score(self, score):
        self.score = score

    def set_selected(self, selected):
        self.selected = selected

    def get_name(self):
        return self.name

    def get_score(self):
        return self.score

    def get_selected(self):
        return self.selected

    def __str__(self):
        return f"Feature: {self.name}, Score: {self.score}, Selected: {self.selected}"

    def __repr__(self):
        return f"Feature({self.name}, {self.score}, {self.selected})"