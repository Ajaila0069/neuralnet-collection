import time

file = open("text.txt", "r").read().strip('\n')

chars = [x for x in file]
charunique = sorted(list(set(chars)))
unique_dict = dict((x, 0) for x in charunique)

class compressor:

    def __init__(self, d):
        self.d = d

    def fork(self, d, k1, k2):
        i1 = d[k1]
        i2 = d[k2]
        compound = (k1, k2)
        added_ind = i1 + i2
        return compound, added_ind

    def branch(self, dictionary):
        current_last = list(dictionary.keys())[-1]
        current_last2 = list(dictionary.keys())[-2]
        new, ind = self.fork(dictionary, current_last, current_last2)
        dictionary[new] = ind
        dictionary = dict((k, v) for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
        del dictionary[current_last]
        del dictionary[current_last2]
        return dictionary

    def tree(self):
        newd = self.branch(self.d)
        while len(newd.keys()) > 1:
            newd = self.branch(newd)
            #time.sleep(0.1)
            print(newd)
        self.d = newd
        self.comptree = list(self.d.keys())[0]
        return self.comptree

    def maxdepth(self, tup, currcode=""):
        if type(tup[0]) == tuple:
            currcode += "0"
            currcode = self.maxdepth(tup[0], currcode)
            return currcode
        if type(tup[0]) == str:
            currcode += "0"
            return len(currcode)

    def getcodes(self, depth):
        codes = []

    """
    def encode(self, chars):
        for char in chars:



    def decode(self, dir, list):
        let = list[dir]
        if let

    def comp(self, tree, text):
        if type(text) == str:
            for char in text
    """

for char in chars:
    unique_dict[char] += 1

unique_dict = dict((k, v) for k, v in sorted(unique_dict.items(), key=lambda item: item[1], reverse=True))

print(unique_dict)

comp = compressor(unique_dict)
tree = comp.tree()
print(comp.maxdepth(tree))
