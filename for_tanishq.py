import os
import threading
import requests

weir = 0

def weeb(largeweeb):
    while True:
        print(requests.Request("www.reddit.com").content)
        print("pls kill me i am eternally depressed and i find no joy in anything anymore. It is really tiring me out, and sometimes i look at the rope i have in my room and think that it would be real nice to just end it with that rope. I just wish life weren't so tiring. I don't want this to go on anymore, and I really want to just finish this whole joke of a life.")
    return True

while weir < 10:
    print(requests.Request("www.reddit.com").text)
    try:
        threading.Thread(target=weeb("no"))
        weir += 1
        print(weir)
    except:
        print("ur computer sucks penis lol")

while True:
    pass
