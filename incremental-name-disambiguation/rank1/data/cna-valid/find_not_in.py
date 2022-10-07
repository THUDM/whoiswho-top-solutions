import json as js

f1=open("./result.v1.json","r",encoding="utf-8")
f2=open("./cna_valid_ground_truth.json","r",encoding="utf-8")

p1=js.load(f1)
p2=js.load(f2)

hit_count=0
miss_count=0
# 对于每一个truth实例，未发现，miss_count+1
# 发现于正确人物，hit_count+1
# 发现于错误人物，wrong_count+1
wrong_count=0

for item in p1:
    