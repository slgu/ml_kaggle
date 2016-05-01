__author__ = 'slgu1'
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

map_feature = {}

map_category_num = {}

feature_names = []
max_values = [142857] * 53
min_values = [142857] * 53
average = [-1] * 53
variance = [-1] * 53
ignore = [False] * 53



def read_feature_data(filename):
    global map_feature
    with open(filename, "r") as f:
        while True:
            str = f.readline()
            if str is None or str.strip() == "":
                break
            arr = str.strip().split(" ")
            feature_name = arr[0]
            map_feature[feature_name] = {}
            if arr[1] == "numeric":
                # numeric value
                map_category_num[feature_name] = -1
                continue
            # get rid of "v" suffix
            category_features = [item[1:] for item in arr[1:]]
            cnt = 0
            s = set()
            for item in category_features:
                map_feature[feature_name][item] = cnt
                cnt += 1
                for word in item.split("_"):
                    s.add(word)
            print feature_name, " word count:", len(s)
            print cnt
            map_category_num[feature_name] = cnt + 1


# feature name 23 idx = 12
# feature name 58 idx = 46
# 5 14 23
# 56 57 58
def vsm(feature_names, vec):
    global max_values
    global min_values
    for idx in range(0, len(vec)):
        feature_name = feature_names[idx]
        if feature_name == "56":
            wuliu = vec[idx]
        elif feature_name == "57":
            wuqi = vec[idx]
        elif feature_name == "58":
            wuba = vec[idx]
        if len(map_feature[feature_name]) == 0:
            # numeric
            vec[idx] = float(vec[idx])
            if max_values[idx] == 142857 or max_values[idx] < vec[idx]:
                max_values[idx] = vec[idx]
            if min_values[idx] == 142857 or min_values[idx] > vec[idx]:
                min_values[idx] = vec[idx]
        else:
            # map index
            vec[idx] = map_feature[feature_name][vec[idx]]
    if wuliu + "_" + wuqi !=  wuba and wuliu != "na" and wuqi != "na" and wuba != "na":
        pass

def read_train_data(filename):
    global map_feature
    global feature_names
    global max_values
    global min_values
    vecs = []
    labels = []
    with open(filename, "r") as f:
        # read feature name
        str = f.readline().strip()
        feature_names = str.split(",")[:-1]
        while True:
            str = f.readline()
            if str is None or str.strip() == "":
                break
            vec = str.strip().split(",")
            label = int(vec[-1])
            if label != -1 and label != 1:
                print("data error")
                exit(-1)
            vec = vec[:-1]
            vsm(feature_names, vec)
            labels.append(label)
            vecs.append(vec)
    return vecs, labels


def dtmodel(depth):
    return DecisionTreeClassifier(max_depth=depth)


def boost_dtmodel(depth, round):
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                              algorithm="SAMME",
                              n_estimators=round)

def forestmodel(round):
    return RandomForestClassifier(n_estimators=round)

def extramodel(round):
    return ExtraTreesClassifier(n_estimators=round, max_depth=None, min_samples_split=1, random_state=0)

def ridge_model(a):
    return Ridge(alpha=a)


def read_test_data(filename):
    global feature_names
    vecs = []
    with open(filename, "r") as f:
        # read feature name
        str = f.readline().strip()
        feature_names = str.split(",")
        while True:
            str = f.readline()
            if str is None or str.strip() == "":
                break
            vec = str.strip().split(",")
            vsm(feature_names, vec)
            vecs.append(vec)
    return vecs


def save(labels, filename):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for idx in range(0, len(labels)):
            f.write("%d,%d\n" % (idx + 1, labels[idx]))


def map_high_dimension(vec):
    global map_category_num
    global map_feature
    global feature_names
    l = len(vec)
    res_vec = []
    for i in range(0, l):
        feature_name = feature_names[i]
        enum_num = map_category_num[feature_name]
        if enum_num == -1:
            # numeric
            res_vec.append(vec[i])
        else:
            tmp = [0] * enum_num
            tmp[vec[i]] = 1
            res_vec += tmp
    return res_vec


def map_high_dimension_arr(vecs):
    res_vecs = []
    l = len(vecs)
    for i in range(0, l):
        res_vecs.append(map_high_dimension(vecs[i]))
    return res_vecs


def get_ignore():
    global max_values
    global min_values
    global ignore
    l = len(max_values)
    for i in range(0, l):
        if max_values[i] != 142857 and max_values[i] == min_values[i]:
            ignore[i] = True


def set_ignore():
    global ignore
    ignore[12] = True
    ignore[46] = True


def ignore_clear(datas):
    l = len(datas)
    dimension = len(datas[0])
    res_datas = []
    for i in range(0, l):
        tmp = []
        for j in range(0, dimension):
            if not ignore[j]:
                tmp.append(datas[i][j])
        res_datas.append(tmp)
    return res_datas


def validate(datas, labels, model):
    predict_labels = model.predict(datas)
    l = len(predict_labels)
    err = 0
    for i in range(0, l):
        if labels[i] != predict_labels[i]:
            err += 1
    print("validate error: %.2f" % (err * 100.0 / l))

if __name__ == "__main__":
    read_feature_data("data/field_types.txt")
    datas, labels = read_train_data("data/data.csv")
    get_ignore()
    datas = ignore_clear(datas)
    test_datas = read_test_data("data/quiz.csv")
    test_datas = ignore_clear(test_datas)
    ada = boost_dtmodel(12, 400)
    forest = forestmodel(200)
    extra = forestmodel(250)
    model = VotingClassifier(estimators=[('ada', ada), ('forest', forest), ('extra', extra)], voting='hard')
    print("begin fit data")
    model.fit(datas, labels)
    print("train model done")
    validate(datas, labels, model)
    predict_labels = model.predict(test_datas)
    save(predict_labels, "data/quiz_out_10.csv")
