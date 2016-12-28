import math


def leave_p_out(features, labels, train_pct):
    """
    Given the features, labels and the percentage of the data that is to be fitted to
    the model, implements the leave-p-out exhaustive cross-validation algorithm to
    split the data into all possible combinations of training/testing samples and
    returns training/testing sets containing each of the training/testing samples
    respectively
    """
    training_set = []  # initialize the training set
    testing_set = []  # initialize the testing set
    # for every index that can be the starting index of a testing sample
    for min_idx in range(len(features[:int(train_pct * len(features))])):
        # find the maximum index for each testing sample
        max_idx = min_idx + math.ceil((1 - train_pct) * len(features))
        features_training_sample = []  # initialize the features training sample
        features_testing_sample = []  # initialize the features testing sample
        labels_training_sample = []  # initialize the labels training sample
        labels_testing_sample = []  # initialize the labels training sample
        for idx in range(len(features)):  # for every index in the features/labels set
            if min_idx <= idx < max_idx:  # if the index is in the range of indices for the testing sample
                features_testing_sample.append(features[idx])
                labels_testing_sample.append(labels[idx])  # append the features and labels to the testing sample
            else:  # if not
                features_training_sample.append(features[idx])
                labels_training_sample.append(labels[idx])  # append the features and labels to the training sample
        # append the samples to the set
        training_set.append(tuple((list(features_training_sample), list(labels_training_sample))))
        testing_set.append(tuple((list(features_testing_sample), list(labels_testing_sample))))
    # append the final sample to the set
    training_set.append(tuple((list(features[:int(len(features) * train_pct)]), list(labels[:int(len(labels) * train_pct)]))))
    testing_set.append(tuple((list(features[int(len(features) * train_pct):]), list(labels[int(len(labels) * train_pct):]))))
    return training_set, testing_set  # return the sets as an unpacked pair


def k_fold(features, labels, k):
    """
    Given the features, labels and the value of k, implements the k-folds non-exhaustive
    cross-validation algorithm to split the data into all possible combinations of
    k-training/testing samples and returns training/testing sets containing each of the
    training/testing samples respectively
    """
    training_set = []  # initialize the training set
    testing_set = []  # initialize the testing set
    # for all indices where one of the k samples start
    for min_idx in range(0, len(features), int(len(features) / k)):
        # calculate the corresponding ending index of the sample
        max_idx = min_idx + int(len(features) / k)
        # use splicing to extract the training and testing samples
        features_training_sample = list(features[:min_idx]) + list(features[max_idx:])
        labels_training_sample = list(labels[:min_idx]) + list(labels[max_idx:])
        # append the samples to the training and testing sets
        training_set.append(tuple((list(features_training_sample), list(labels_training_sample))))
        testing_set.append(tuple((list(features[min_idx:max_idx]), list(labels[min_idx:max_idx]))))
    return training_set, testing_set  # return the training and the testing sets as an unpacked pair


