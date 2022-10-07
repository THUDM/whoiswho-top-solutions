from tool.interface import MatchName

import json


def load_data():
    with open("./data/test_data.json",
              "r",
              encoding="utf-8") as f:
        datas = json.load(f)

    return datas


if __name__ == "__main__":
    datas = load_data()

    i = 0

    print("loose match: ")
    for data in datas:
        res = MatchName(data["name"], data["names"], True)
        print(res)
        # if len(res[1]) > 0:
        #     i += 1
        #     if i < 300:
        #         continue
        #     print(i, "=========", res)

        #     if i >= 500:

        #         break
    # print("strict match: ")
    # for data in datas:
    #     res = MatchName(data["name"], data["names"], False)
    #     print(res)