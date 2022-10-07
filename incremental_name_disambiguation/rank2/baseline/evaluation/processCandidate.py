import sys 
# sys.path.append('..')
from os.path import join, abspath, dirname
sys.path.append(join(abspath(dirname(__file__)), ".."))
import json
from collections import defaultdict
from tqdm import tqdm
from character.name_match.tool.interface import FindMain



def printInfo(dicts):
    aNum = 0
    pNum = 0
    for name, aidPid in dicts.items():
        aNum += len(aidPid)
        for aid, pids in aidPid.items():
            pNum += len(pids)

    print("#Name %d, #Author %d, #Paper %d"%(len(dicts), aNum, pNum))


def findMainAuthorIndex(pros, prosInfo, nameAidPid):
    printInfo(nameAidPid)

    # Find the main author index for each paper
    keyNames = list(nameAidPid.keys())
    newIndex = {}
    errCount = 0
    for i in tqdm(range(len(keyNames))):
        tmpName = defaultdict(list)
        name = keyNames[i]
        aidPid = nameAidPid[name]
        for aid, pids in aidPid.items():
            tmpPubs = []
            for each in pids:
                coauthors = [tmp["name"] for tmp in prosInfo[each]["authors"]]
                res = FindMain(name, coauthors)
                try:
                    newPid = each + '-' + str(res[0][0][1])
                    tmpPubs.append(newPid)
                except:
                    # print(name, coauthors, res)
                    errCount += 1

            tmpName[aid] = tmpPubs
        newIndex[name] = tmpName

    printInfo(newIndex)
    print("errCount: ", errCount)
    return newIndex


def saveMainAuthorIndex(prosFile, prosInfoFile, savePath):
    with open(prosFile, 'r',encoding='utf-8') as files:
        pros = json.load(files)

    with open(prosInfoFile, 'r',encoding='utf-8') as files:
        prosInfo = json.load(files)

    if 'whole' in prosFile:
        # Merge all authors under the same name.
        nameAidPid = defaultdict(dict)
        for aid, info in pros.items():
            name = info["name"]
            pubs = info["pubs"]
            nameAidPid[name][aid] = pubs
    else:
        nameAidPid = pros

    newIndex = findMainAuthorIndex(pros, prosInfo, nameAidPid)
    with open(savePath, 'w') as files:
        json.dump(newIndex, files, indent=4, ensure_ascii = False)


def findCandidates(nameAidPidFile, data_types):
    with open(nameAidPidFile, 'r',encoding='utf-8') as files:
        nameAidPid = json.load(files)

    dataDir = './datas/Task1/cna-{}/'.format(data_types)
    with open(dataDir + "cna_{}_unass.json".format(data_types), 'r',encoding='utf-8') as files:
        validUnass = json.load(files)

    with open(dataDir + "cna_{}_unass_pub.json".format(data_types), 'r',encoding='utf-8') as files:
        validUnassPub = json.load(files)

    candiNames = list(nameAidPid.keys())
    print("#Unass: %d #candiNames: %d" % (len(validUnass), len(candiNames)))

    unassCandi = []
    notMatch = 0
    for each in tqdm(validUnass):
        pid, index = each.split('-')
        mainName = validUnassPub[pid]["authors"][int(index)]["name"]
        res = FindMain(mainName, candiNames)
        try:
            # newPid = each + '-' + str(res[0][0][0])
            unassCandi.append((each, res[0][0][0]))
            # print(mainName, res)
        except:
            notMatch += 1
        # exit()
    print("Matched: %d Not Match: %d" % (len(unassCandi), notMatch))
    with open("./datas/{}_unassCandi.json".format(data_types), 'w') as files:
        json.dump(unassCandi, files, indent=4, ensure_ascii=False)


# ================= whole ========================
prosFile = './datas/Task1/cna-valid/whole_author_profiles.json'
prosInfoFile = './datas/Task1/cna-valid/whole_author_profiles_pub.json'
savePath = './datas/proNameAuthorPubs.json'
saveMainAuthorIndex(prosFile, prosInfoFile, savePath)

# ================== train =====================
prosFile = './datas/train_author_profile.json'
prosInfoFile = './datas/Task1/train/train_pub.json'
savePath = './datas/train_proNameAuthorPubs.json'
saveMainAuthorIndex(prosFile, prosInfoFile, savePath)

# ================== test =====================
prosFile = './datas/test_author_profile.json'
prosInfoFile = './datas/Task1/train/train_pub.json'
savePath = './datas/test_proNameAuthorPubs.json'
saveMainAuthorIndex(prosFile, prosInfoFile, savePath)


# Find candidates for unass papers
nameAidPidFile = './datas/proNameAuthorPubs.json'
findCandidates(nameAidPidFile, 'valid')
findCandidates(nameAidPidFile, 'test')

