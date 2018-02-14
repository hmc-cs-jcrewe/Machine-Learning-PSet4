"""
Author      : Jackson Crewe & Matt Guillory
Class       : HMC CS 158
Date        : 2018 Feb 14
Description : Perceptron vs Logistic Regression on a Phoneme Dataset
"""

# utilities
from util import *

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

######################################################################
# functions
######################################################################

def cv_performance(clf, train_data, kfs) :
    """
    Determine classifier performance across multiple trials using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kfs        -- array of size n_trials
                      each element is one fold from model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_trials, n_fold)
                      each element is the (accuracy) score of one fold in one trial
    """

    n_trials = len(kfs)
    n_folds = kfs[0].n_splits
    scores = np.zeros((n_trials, n_folds))

    ### ========== TODO : START ========== ###
    for i in range(n_trials):
        scores[i] = cv_performance_one_trial(clf, train_data, kfs[i])

    ### ========== TODO : END ========== ###

    return scores


def cv_performance_one_trial(clf, train_data, kf) :
    """
    Compute classifier performance across multiple folds using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kf         -- model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_fold, )
                      each element is the (accuracy) score of one fold
    """

    scores = np.zeros(kf.n_splits)

    ### ========== TODO : START ========== ###
    index = 0
    for train_index, test_index in kf.split(train_data.X, train_data.y):
        X_train, X_test = train_data.X[train_index], train_data.X[test_index]
        y_train, y_test = train_data.y[train_index], train_data.y[test_index]
        clf.fit(X_train, y_train)
        scores[index] = clf.score(X_test, y_test)
        index += 1

    ### ========== TODO : END ========== ###

    return scores


######################################################################
# main
######################################################################

def main() :
    np.random.seed(1234)

    #========================================
    # load data
    train_data = load_data("phoneme_train.csv")

    ### ========== TODO : START ========== ###
    clf = Perceptron(max_iter = 1000)
    clf.fit(train_data.X, train_data.y)
    print 'score = %d' % clf.score(train_data.X, train_data.y)

    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###

    # create model_selection.KFold for computing classifier performance
    kfs = [model_selection.KFold(n_splits=10, shuffle=True) for i in range(10)]

    # computes classifier performance for each classifier
    clf1 = Perceptron(max_iter = 1000)
    perceptron_scores = cv_performance(clf1, train_data, kfs)
    perceptron_scores = perceptron_scores.flatten()
    print perceptron_scores

    clf2 = DummyClassifier(strategy = 'most_frequent')
    dummy_scores = cv_performance(clf2, train_data, kfs)
    dummy_scores = dummy_scores.flatten()
    print dummy_scores

    clf3 = LogisticRegression(C = 1e5)
    logistic_scores = cv_performance(clf3, train_data, kfs)
    logistic_scores = logistic_scores.flatten()
    print logistic_scores

    perceptron_mean = np.mean(perceptron_scores, dtype=np.float64)
    perceptron_sdev = np.std(perceptron_scores, dtype=np.float64)

    dummy_mean = np.mean(dummy_scores, dtype=np.float64)
    dummy_sdev = np.std(dummy_scores, dtype=np.float64)

    logistic_mean = np.mean(logistic_scores, dtype=np.float64)
    logistic_sdev = np.std(logistic_scores, dtype=np.float64)

    print 'the perceptron mean is: %03.3f. The standard deviation is: %03.3f' % (perceptron_mean, perceptron_sdev)
    print 'the dummy mean is: %03.3f. The standard deviation is: %03.3f' % (dummy_mean, dummy_sdev)
    print 'the logistic mean is: %03.3f. The standard deviation is: %03.3f' % (logistic_mean, logistic_sdev)

    X_scaled = preprocessing.scale(train_data.X)
    train_scaled = Data(X_scaled, train_data.y)

    # computes classifier performance for each classifier
    clf1 = Perceptron(max_iter = 1000)
    perceptron_scores_scaled = cv_performance(clf1, train_scaled, kfs)
    perceptron_scores_scaled = perceptron_scores_scaled.flatten()
    print perceptron_scores_scaled

    clf2 = DummyClassifier(strategy = 'most_frequent')
    dummy_scores_scaled = cv_performance(clf2, train_scaled, kfs)
    dummy_scores_scaled = dummy_scores_scaled.flatten()
    print dummy_scores_scaled

    clf3 = LogisticRegression(C = 1e5)
    logistic_scores_scaled = cv_performance(clf3, train_scaled, kfs)
    logistic_scores_scaled = logistic_scores_scaled.flatten()
    print logistic_scores_scaled

    perceptron_mean_scaled = np.mean(perceptron_scores_scaled, dtype=np.float64)
    perceptron_sdev_scaled = np.std(perceptron_scores_scaled, dtype=np.float64)

    dummy_mean_scaled = np.mean(dummy_scores_scaled, dtype=np.float64)
    dummy_sdev_scaled = np.std(dummy_scores_scaled, dtype=np.float64)

    logistic_mean_scaled = np.mean(logistic_scores_scaled, dtype=np.float64)
    logistic_sdev_scaled = np.std(logistic_scores_scaled, dtype=np.float64)

    print 'the perceptron mean_scaled is: %03.3f. The standard deviation is: %03.3f' % (perceptron_mean_scaled, perceptron_sdev_scaled)
    print 'the dummy mean_scaled is: %03.3f. The standard deviation is: %03.3f' % (dummy_mean_scaled, dummy_sdev_scaled)
    print 'the logistic mean_scaled is: %03.3f. The standard deviation is: %03.3f' % (logistic_mean_scaled, logistic_sdev_scaled)


    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part e: plot

    ### ========== TODO : END ========== ###

if __name__ == "__main__" :
    main()
