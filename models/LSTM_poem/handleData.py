import json
paragraphs = []
with open('./data/all.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
    paragraphs = json_data["paragraphs"]

sentences = []
for key,value in paragraphs:
    for sen in value:
        sentences.append(sentences)

print(111)

