from tool.util import match_name_two, match_name_one, match_name_three, match_name_four, match_name_five, match_name_six, match_name_seven
from tool.is_chinese import cleaning_name

if __name__ == "__main__":
    data = ("jie tang", "Jie T")

    res = match_name_one(cleaning_name(data[0]), cleaning_name(data[1]), True)
    print("match_name_one:  ", res)
    res = match_name_two(cleaning_name(data[0]), cleaning_name(data[1]), True)
    print("match_name_two:  ", res)

    res = match_name_three(cleaning_name(data[0]), cleaning_name(data[1]),
                           True)
    print("match_name_three:  ", res)
    res = match_name_four(cleaning_name(data[0]), cleaning_name(data[1]), True)
    print("match_name_four:  ", res)
    res = match_name_five(cleaning_name(data[0]), cleaning_name(data[1]), True)
    print("match_name_five:  ", res)
    res = match_name_six(cleaning_name(data[0]), cleaning_name(data[1]), True)
    print("match_name_six:  ", res)
    res = match_name_seven(cleaning_name(data[0]), cleaning_name(data[1]),
                           True)
    print("match_name_seven:  ", res)

    print("match res: ", res)