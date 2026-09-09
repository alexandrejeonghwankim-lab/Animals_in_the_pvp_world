import json


with open('test_50.json', 'r') as file:
    data = json.load(file)


res = data["response"]["apps"]
print(type(res))
