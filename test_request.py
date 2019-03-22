import requests
r = requests.get(r"https://api.github.com/users/acombs/starred")
r.json()