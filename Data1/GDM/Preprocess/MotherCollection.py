
class MotherCollection:
    def __init__(self):
        self.mother_list = []

    def add_mom(self, mom):
        self.mother_list.append(mom)

    def add_moms(self, moms_list):
        for mom in moms_list:
            self.add_mom(mom)

    def return_positive(self):
        count = 0
        for mom in self.mother_list:
            if mom.sick_or_not == 0:
                count += 1
        return count
