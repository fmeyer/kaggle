from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
import csv_io as csv
import llfun as logloss
import numpy as np

def main():
    #read in  data, parse into training and target sets
    train = csv.read_data("../Data/train.csv")
    target = np.array( [x[0] for x in train] )
    train = np.array( [x[1:280] for x in train] )

    #In this case we'll use a random forest, but this could be any classifier
    cfr = RandomForestClassifier(n_estimators=120, min_samples_split=2, n_jobs=-1, max_depth=None) #.46
    #cfr = GradientBoostingClassifier(n_estimators=120, learn_rate=0.57, max_depth=1) #.50
    #cfr = ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_split=1) #.489

    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(train), k=5, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    count = 0
    for traincv, testcv in cv:
        probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        result = logloss.llfun(target[testcv], [x[1] for x in probas])
        count += 1
        print('fold: %d, result: %f' % (count, result))
        results.append( result )

    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )

    test = csv.read_data("../Data/test.csv")
    predicted_probs = cfr.predict_proba( [x[0:279] for x in test])
    predicted_probs = ["%f" % x[1] for x in predicted_probs]
    csv.write_delimited_file("../Submissions/rf_cv.csv",
                                predicted_probs)

if __name__=="__main__":
    main()
