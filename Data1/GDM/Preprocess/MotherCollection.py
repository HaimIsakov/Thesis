
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

    def return_ids(self):
        ids_list = []
        for mom in self.mother_list:
            ids_list.append(mom.id)
        return ids_list

    def value_counts_trimesters(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count_other = 0
        for mom in self.mother_list:
            if mom.trimester == '1':
                count1 += 1
            elif mom.trimester == '2':
                count2 += 1
            elif mom.trimester == '3':
                count3 += 1
            else:
                count_other += 1
        return count1, count2, count3, count_other

    def __len__(self):
        return len(self.mother_list)