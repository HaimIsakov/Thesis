
class Mother:
    def __init__(self, id, trimester, repetition, sick_or_not, test_way='stool'):
        self.id = id
        self.trimester = trimester
        self.repetition = repetition
        self.sick_or_not = sick_or_not
        self.test_way = test_way


    def to_string(self):
        return f"Id:{self.id}, Trimester:{self.trimester}, Repetition:{self.repetition}, Way-of-Test:{self.test_way}" \
               f" Sick: {self.sick_or_not}"
