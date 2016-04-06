from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# sex,city,build,race,pct,pos_knife,pos_handgun,pos_rifle,pos_assault,pos_machgun,
# pos_otherweap,pos_illegal,cs_susp_obj,rf_unseasonal_attire,cs_crime_attire,cs_susp_bulge,
# cs_match_desc,cs_recon,cs_lookout,cs_drug_trade,cs_covert,rf_violent,cs_violent,ac_crime_area,ac_crime_time,
# ac_crime_assoc,ac_avoid_cops,label
# 27 features
variance = [2, 5, 4, 8, 78] + [2] * 22


def read_data(filename):
    desc = []
    datas = []
    with open(filename, "r") as f:
        # first read description
        str = f.readline().strip()
        desc = str.split()
        while True:
            str = f.readline()
            if str is None or str == "" or str.strip() == "":
                break
            vec = str.strip().split(",")
            # ignore Z data
            if "Z" in vec:
                continue
            # deal with sex
            if vec[0] == "M":
                vec[0] = 0
            else:
                vec[0] = 1
            # deal with city
            vec[1] = int(vec[1][5:]) - 1
            # deal with build
            vec[2] = int(vec[2][6:]) - 1
            # deal with race
            vec[3] = int(vec[3][5:]) - 1
            # deal with pct
            vec[4] = int(vec[4][4:])
            vec = [int(item) for item in vec]
            datas.append(vec)
    return datas


def enhance(datas, rate):
    xs = []
    ys = []
    for vec in datas:
        if vec[-1] == 1:
            for i in range(0, rate):
                xs.append(vec[0:-1])
                ys.append(vec[-1])
        else:
            xs.append(vec[0:-1])
            ys.append(vec[-1])
    return xs, ys


def split(datas):
    xs = []
    ys = []
    for vec in datas:
        xs.append(vec[0:-1])
        ys.append(vec[-1])
    return xs, ys


# to prevent -1 data
# just first 4 columns
def map_to_high_dimension(vec):
    num_of_colum = 5
    v = []
    for i in range(0, num_of_colum):
        tmp = [0] * (variance[i] + 1)
        tmp[vec[i]] = 1
        v.append(tmp)
    res = []
    for i in range(0, num_of_colum):
        res += v[i]
    res += vec[num_of_colum:]
    return res


def map_to_high_dimension_arr(datas):
    l = len(datas)
    res = []
    for i in range(0, l):
        res.append(map_to_high_dimension(datas[i]))
    return res


def boostdecision():
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),
                              algorithm="SAMME",
                              n_estimators=80)


def decision():
    return DecisionTreeClassifier(max_depth=6)


def read_test_data(filename):
    datas = []
    with open(filename, "r") as f:
        # first read description
        f.readline().strip()
        while True:
            str = f.readline()
            if str is None or str == "" or str.strip() == "":
                break
            vec = str.strip().split(",")
            for i in range(len(vec)):
                if vec[i] == "Z":
                    vec[i] = -1

            # ignore Z data
            if "Z" in vec:
                continue
            # deal with sex
            if vec[0] == "M":
                vec[0] = 0
            elif vec[0] == "F":
                vec[0] = 1
            # deal with city
            if vec[1] != -1:
                vec[1] = int(vec[1][5:]) - 1
            # deal with build
            if vec[2] != -1:
                vec[2] = int(vec[2][6:]) - 1
            # deal with race
            if vec[3] != -1:
                vec[3] = int(vec[3][5:]) - 1
            # deal with pct
            if vec[4] != -1:
                vec[4] = int(vec[4][4:])
            vec = [int(item) for item in vec]
            datas.append(vec)
    return datas


# generate all possible combination of -1
def gen(vec, i):
    global variance
    if len(vec) == 0:
        return [[]]
    datas = gen(vec[1:], i + 1)
    res = []
    for data in datas:
        if vec[0] == -1:
            # all combination
            for j in range(0, variance[i]):
                res.append([j] + data)
        else:
            res.append([vec[0]] + data)
    return res


def test_with_rate_avg(model, datas):
    # return labels
    l = len(datas)
    labels = []
    for i in range(0, l):
        gen_datas = gen(datas[i], 0)
        res = model.predict_proba(gen_datas)
        len_choice = len(res)
        prob_not_arrest = 0
        for j in range(0, len_choice):
            prob_not_arrest += res[j][0]
        prob_not_arrest /= len_choice
        if prob_not_arrest > (1 - prob_not_arrest) * 10:
            labels.append(-1)
        else:
            labels.append(1)
    return labels


def test_with_rate(model, datas):
    # return labels
    l = len(datas)
    labels = []
    for i in range(0, l):
        prob_not_arrest = model.predict_proba(datas[i])[0][0]
        if prob_not_arrest > (1 - prob_not_arrest) * 10:
            labels.append(-1)
        else:
            labels.append(1)
    return labels


def test_with_avg(model, datas):
    l = len(datas)
    labels = []
    for i in range(0, l):
        gen_datas = gen(datas[i], 0)
        res = model.predict_proba(gen_datas)
        len_choice = len(res)
        prob_not_arrest = 0
        for j in range(0, len_choice):
            prob_not_arrest += res[j][0]
        prob_not_arrest /= len_choice
        if prob_not_arrest > 0.5:
            labels.append(-1)
        else:
            labels.append(1)
    return labels


def test(model, datas):
    return model.predict(datas)


def save_file(labels, filename):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        l = len(labels)
        for i in range(0, l):
            f.write("%d,%d\n" % (i + 1, labels[i]))


def main():
    print "begin read data"
    datas = read_data("data/data.csv")
    test_datas = read_test_data("data/quiz.csv")
    # enhance 1 data 10 times
    xs, ys = enhance(datas, 10)
    # map_to_high_dimension
    xs = map_to_high_dimension_arr(xs)
    test_datas = map_to_high_dimension_arr(test_datas)
    print len(xs)
    print len(xs[0])
    model = boostdecision()
    print "begin training"
    model.fit(xs, ys)
    print "training done"
    # begin validate
    # begin test
    print "begin test"
    labels = test(model, test_datas)
    print  "begin write to file"
    save_file(labels, "data/quiz_out_8.csv")
    print "all done"


if __name__ == "__main__":
    main()
