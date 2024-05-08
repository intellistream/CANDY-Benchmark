# Weapons Updated until 2024-03-23
import os


def load_list_from_file(filename):
    with open(filename, 'r') as file:
        string_list = file.readlines()
        # Remove trailing newline characters
        string_list = [string.strip() for string in string_list]
        return string_list


def load_file_as_str(filename):
    with open(filename, 'r') as file:
        file_contents = file.read()
        return file_contents


def paraseInCategories(categories, prefix="warthunder2/"):
    keyListRu = []
    descListRu = []
    usageListRu = []
    historyListRu = []
    for category in categories:
        fname = category
        fnameKey = prefix + "data/" + fname + "/keys" + ".txt"
        keyList = load_list_from_file(fnameKey)
        keyListRu = keyListRu + keyList
        for i in range(len(keyList)):
            strI = str(i + 1)
            fnameDesc = prefix + "data/" + fname + "/description_" + strI + ".txt"
            fnameUsage = prefix + "data/" + fname + "/usage_" + strI + ".txt"
            fnameHistory = prefix + "data/" + fname + "/history_" + strI + ".txt"
            ctxI = load_file_as_str(fnameDesc)
            descListRu = descListRu + [ctxI]
            ctxI = load_file_as_str(fnameUsage)
            usageListRu = usageListRu + [ctxI]
            ctxI = load_file_as_str(fnameHistory)
            historyListRu = historyListRu + [ctxI]
    return keyListRu, descListRu, usageListRu, historyListRu


def genRencentCategories():
    countries = [
        'USA',
        'Germany',
        'USSR',
        'Britain',
        'Japan',
        'China',
        'Italy',
        'France',
        'France',
        'Sweden',
        'Israel'
    ]
    types = [
        'aircraft',
        'ground_vehicles'
    ]
    categories = []
    for i in countries:
        for j in types:
            tempStr = i + '_' + j
            categories.append(tempStr)
    return categories


if __name__ == "__main__":
    countries = [
        'USA',
        'Germany',
        'USSR',
        'Britain',
        'Japan',
        'China',
        'Italy',
        'France',
        'France',
        'Sweden',
        'Israel'
    ]
    types = [
        'aircraft',
        'ground_vehicles'
    ]
    categories = []
    for i in countries:
        for j in types:
            tempStr = i + '_' + j
            categories.append(tempStr)
    print(categories)
    keys, descs, usages, histories = paraseInCategories(categories, "")
    desMap = {k: v for k, v in zip(keys, descs)}
    usageMap = {k: v for k, v in zip(keys, usages)}
    historyMap = {k: v for k, v in zip(keys, histories)}
    for i in keys:
        print(usageMap[i])
