import json
import random

keys = [tuple(i) for i in json.load(open("lol.json", "r"))]
total = dict((str(k), None) for k in keys)
ref = open("data.json", "w")

random.shuffle(keys)
for i in keys:
    association = input(f"Are the words \"{i[0]}\" and \"{i[1]}\" highly associated?: ")
    total.update({str(i) : str(association)})
    json.dump(total, ref)
