from __future__ import print_function

import os
from nolearn.convnet import ConvNetFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from sklearn.svm import LinearSVC

from sklearn.cross_validation import KFold

DECAF_IMAGENET_DIR = '/disks/research/Datasets/imagenet/'
TRAIN_DATA_DIR = '/disks/research/Datasets/train/'
PREDICT_DATA_DIR = '/disks/research/Datasets/test1/'

import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

#alist.sort(key=natural_keys)
#print(alist)


def get_testdataset():
    files = [PREDICT_DATA_DIR + fn for fn in os.listdir(PREDICT_DATA_DIR)]
    files.sort(key=natural_keys)
    return files


def get_dataset():
    cat_dir = TRAIN_DATA_DIR + 'cat/'
    cat_filenames = [cat_dir + fn for fn in os.listdir(cat_dir)]
    dog_dir = TRAIN_DATA_DIR + 'dog/'
    dog_filenames = [dog_dir + fn for fn in os.listdir(dog_dir)]

    labels = [0] * len(cat_filenames) + [1] * len(dog_filenames)
    filenames = cat_filenames + dog_filenames
    return shuffle(filenames, labels, random_state=0)


def main():
    convnet = ConvNetFeatures(
        pretrained_params=DECAF_IMAGENET_DIR + 'imagenet.decafnet.epoch90',
        pretrained_meta=DECAF_IMAGENET_DIR + 'imagenet.decafnet.meta',
        )

    clf = LogisticRegression()

    clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
    clf = KNeighborsClassifier(n_neighbors=15)

    clf = LinearSVC(penalty='l1', loss='l2', dual=False, tol=1e-3)

    pl = Pipeline([
        ('convnet', convnet),
        ('clf', clf),
        ])

    X, y = get_dataset()
    print("Total Dataset size:", len(X))
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[15000:15600], y[15000:15600]

    Z = get_testdataset()

    cv = KFold(len(X_train), n_folds=2, indices=True)

    for traincv, testcv in cv:
        print("traincv", len(traincv), "testcv", len(testcv))
        print("Fitting...")
        pl.fit(X_train[traincv], y_train[traincv])
        print("Predicting...")
        y_pred = pl.predict(X_train[testcv])
        print("Accuracy: %.3f" % accuracy_score(y_train[testcv], y_pred))

    #print("Fittin...")
    #pl.fit(X_train, y_train)
    #print("Predicting...")
    #y_pred = pl.predict(X_test)
    #print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

    predict = []

    for i in range(50):
        #import pdb
        #pdb.set_trace()

        s = i * 250
        e = s + 250

        print("batch {0}, source: {1}, end:{2}".format(i, s, e))
        batch = pl.predict(Z[s:e])
        #z_pred = pl.predict(Z[:200])
        predict.extend(batch)

    id = 1
    f = open('submission_svm.txt', 'w')
    f.write("id,label\n")

    for item in predict:
        f.write("{0},{1}\n".format(id, item))
        id = id + 1

    f.close()

    #import pdb
    #pdb.set_trace()


main()
