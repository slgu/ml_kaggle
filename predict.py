__author__ = 'slgu1'
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

map_feature = {}

map_category_num = {}

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
                continue
            # get rid of "v" suffix
            category_features = [item[1:] for item in arr[1:]]
            cnt = 0
            for item in category_features:
                map_feature[feature_name][item] = cnt
                cnt = cnt + 1

def vsm(feature_names, vec):
    for idx in range(0, len(vec)):
        feature_name = feature_names[idx]
        if len(map_feature[feature_name]) == 0:
            # numeric
            vec[idx] = float(vec[idx])
            pass
        else:
            # map index
            vec[idx] = map_feature[feature_name][vec[idx]]


def read_train_data(filename):
    global map_feature
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


def read_test_data(filename):
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


def map_high_dimension(vecs):


if __name__ == "__main__":
    read_feature_data("data/field_types.txt")
    datas, labels = read_train_data("data/data.csv")
    test_datas = read_test_data("data/quiz.csv")
    model = boost_dtmodel(8, 200)
    print("begin fit data")
    model.fit(datas, labels)
    print("train model done")
    predict_labels = model.predict(test_datas)
    save(predict_labels, "data/quiz_out.csv")
    # just run a raw decision tree