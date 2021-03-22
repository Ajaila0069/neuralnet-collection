from nltk import RegexpTokenizer

r = RegexpTokenizer(r'\w+')

plntxt = open("plaintext.txt", "r").read().lower()

words = r.tokenize(plntxt)
#print(words)

alphabet = "abcdefghijklmnopqrstuvwxyz"
cypheralpha = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", ".-", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]

cypherdict = dict((char, char2) for char, char2 in zip(alphabet, cypheralpha))

cytxt = ""
for word in words:
    for char in word:
        cytxt += cypherdict[char]
        cytxt += " "
    cytxt += "   "

print(cytxt)
