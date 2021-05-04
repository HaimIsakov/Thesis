import re


class Bacteria:
    def __init__(self, string, val):
        lst = re.split("; |__| ", string)
        self.val = val
        # removing letters and blank spaces
        for i in range(0, len(lst)):
            if len(lst[i]) < 2:
                lst[i] = 0
        lst = [value for value in lst if value != 0]
        self.lst = lst
