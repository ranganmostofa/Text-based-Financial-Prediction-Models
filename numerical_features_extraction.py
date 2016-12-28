import math
import numpy as np


def array_to_list(array_matrix):
    """
    Given a matrix built using numpy arrays, builds the same matrix using
    Python lists and returns this matrix
    """
    list_matrix = []  # initialize empty Python list
    for row in array_matrix:  # for every row in the numpy matrix
        # convert the numpy row to a Python list and add it to the list
        # initialized above
        list_matrix.append(list(row))
    return list_matrix  # return the matrix


def list_to_array(list_matrix):
    """
    Given a matrix built using Python lists, builds the same matrix using
    numpy arrays and returns this matrix
    """
    array_matrix = []  # initialize empty Python list
    for list_row in list_matrix:  # for every row in the Python matrix
        # convert the row represented as a Python list to a numpy array
        array_row = np.array(list_row)
        # add the numpy array to the matrix
        array_matrix.append(array_row)
    # convert the outer Python list to a numpy array and return this matrix
    return np.array(array_matrix)


def append_numerical_features(textual_features_matrix, numerical_features_matrix):
    """
    Given a textual and numerical features matrix, appends the two and returns
    the numpy array implementation of the resulting matrix
    """
    # make a copy of the input numerical features matrix
    numerical_features_matrix_copy = list_to_array(numerical_features_matrix)
    # concatenate the textual and numerical features matrices and return the numpy
    # array implementation of the resulting matrix
    return np.concatenate((textual_features_matrix, numerical_features_matrix_copy), axis=1)


def compute_trend(percentage_changes):
    """
    Given the percentage changes in daily stock prices represented as a list of
    floats, returns a list of trends using the following scheme:
    * -1: if the percentage change is negative
    * 0: if the percentage change is 0
    * 1: if the percentage change is positive
    """
    trends = []  # initialize an empty list
    for change in percentage_changes:  # for every percentage change
        if change != 0.00:  # if the change is not 0
            trends.append(change / abs(change))  # map to -1 or 1 accordingly
        else:  # otherwise
            trends.append(0)  # map to 0
    return trends  # return the list of trends


def compute_num_pos(trends):
    """
    Given a list of trends (-1, 0 or 1) in daily stock prices, return the number
    of times there was a positive change in the prices
    """
    count = 0  # initialize the counter to 0
    for trend in trends:  # for every trend
        if trend == 1:  # if the trend is 1
            count += 1  # increase the counter by 1
    return count  # return the value stored by the counter


def compute_stepwise_num_pos(trends):
    """
    Given a list of trends (-1, 0 or 1) in daily stock prices, returns a list of
    stepwise count of positive trends
    """
    stepwise_num_pos = []  # initialize empty list
    for idx in range(len(trends)):  # for every trend
        # get a sublist of trends up to the current index
        trends_sublist = trends[:idx + 1]
        # compute the number of positive trends in the sublist and append this
        # value to the list initialized above
        stepwise_num_pos.append(compute_num_pos(trends_sublist))
    return stepwise_num_pos  # return the list of stepwise count of positive trends


def get_last_pos(percentage_changes):
    """
    Given a list of percentage changes, returns the distance between the occurrence
    of a positive trend and the last trend in the input
    """
    trends = compute_trend(percentage_changes)  # compute trends
    for idx in range(len(trends) - 1, -1, -1):  # for every trend starting from the last one
        if trends[idx] == 1:  # if the trend is positive
            # return the distance of this positive trend from the trend succeeding the last one
            return len(trends) - idx
    return -1  # if there are no positive trends, return -1


def classify_magnitude(percentage_changes):
    """
    Given a list of percentage changes, returns a list of integers that classify
    the input percentage changes
    """
    classified_changes = []  # initialize an empty list of classified percentage changes
    for percentage_change in percentage_changes:  # for every percentage change
        # classify the percentage change according to its magnitude, using the integers
        # from -6 to 6
        # The scheme is outlined in the code below
        if percentage_change < -2.50:
            classified_changes.append(-6)
        elif -2.50 <= percentage_change < -2.00:
            classified_changes.append(-5)
        elif -2.00 <= percentage_change < -1.50:
            classified_changes.append(-4)
        elif -1.50 <= percentage_change < -1.00:
            classified_changes.append(-3)
        elif -1.00 <= percentage_change < -0.50:
            classified_changes.append(-2)
        elif -0.50 <= percentage_change < 0.00:
            classified_changes.append(-1)
        elif percentage_change == 0.00:
            classified_changes.append(0)
        elif 0.00 < percentage_change <= 0.50:
            classified_changes.append(1)
        elif 0.50 < percentage_change <= 1.00:
            classified_changes.append(2)
        elif 1.00 < percentage_change <= 1.50:
            classified_changes.append(3)
        elif 1.50 < percentage_change <= 2.00:
            classified_changes.append(4)
        elif 2.00 < percentage_change <= 2.50:
            classified_changes.append(5)
        elif percentage_change > 2.50:
            classified_changes.append(6)
    return classified_changes  # return the list of classified changes


def get_ngrams(ngram_size, iterable):
    """
    Given the size of the n-grams and an iterable, returns a list of tuples of
    the input size using the values from the input iterable
    """
    ngrams = []  # initialize an empty list of tuple/n-grams
    # for every value in the iterable with at least ngram_size - 1 number of
    # succeeding elements
    for idx in range(len(iterable) - ngram_size + 1):
        # create a tuple including the value positioned at the current index and
        # the ngram_size - 1 succeeding values
        ngrams.append(tuple(iterable[idx:idx + ngram_size]))
    return ngrams  # return the list of tuples


def to_decimal(binary):
    """
    Given a binary number represented as a list of 1s and -1s, returns the
    corresponding decimal number
    """
    decimal = 0  # initialize the decimal number to 0
    for idx in range(len(binary)):  # for every digit in the binary number
        if binary[idx] == -1:  # map all -1s to 0s
            binary[idx] = 0
        # update the decimal counter
        decimal += binary[idx] * math.pow(2, len(binary) - idx - 1)
    return decimal  # return the value of the decimal


def to_feature_vector(binary):
    """
    Given a binary number represented as a list of 1s and -1s, returns the
    feature vector corresponding to the binary number
    """
    decimal = to_decimal(binary)  # convert the binary number to a decimal number
    # calculate the size of the feature vector
    vector_size = int(math.pow(2, len(binary)))
    # preallocate the feature vector as a Pyhton list containing zeros only
    feature_vector = [0] * vector_size
    # access the correct index position and flip the bit to a 1 while others
    # remain 0
    feature_vector[vector_size - int(decimal) - 1] = 1
    return feature_vector  # return the feature vector


def classify_ngrams(ngrams):
    """
    Given a list of n-grams, creates a vector of integer where each integer
    refers to an unique n-gram
    """
    ngram_vector = []  # initialize an empty list of integers
    for ngram in ngrams:  # for every n-gram in the input list of n-grams
        # compute the corresponding integer and append it to the list initialized above
        ngram_vector.append(to_feature_vector(list(ngram)).index(1) + 1)
    ngram_vector.reverse()  # reverse the list
    return ngram_vector  # return the list of integers


def compute_mean(data):
    """
    Given a list of values, computes and returns the mean of the values
    """
    return float(sum(data)) / len(data)  # compute and return the mean


def compute_exponentially_weighted_mean(data):
    """
    Given a list of values, computes the exponentially weighted mean (weighted
    with e = 2.713) of the values, where the higher the index the lower the weight
    """
    total_num = 0.00  # initialize the numerator and the denominator
    total_den = 0.00
    for idx in range(len(data)):  # for every datum
        # update the numerator and the denominator
        total_num += data[idx] * math.pow(math.e, len(data) - idx - 1)
        total_den += math.pow(math.e, len(data) - idx - 1)
    return total_num / total_den  # compute and return the average


def numerical_features(stock_long, stock_short, stock_signal, stock_n_trends, stock_ngram_size, index_long, index_short, index_signal, index_n_trends, index_ngram_size, date_pairs, percentage_changes, stock_prices, index_percentage_changes, index_prices):
    """
    Creates and returns the numerical features matrix, new list of date pairs and
    new percentage changes since some data points had to be trimmed due to the size
    of the variable n below
    """
    new_date_pairs = []  # initialize empty lists
    new_percentage_changes = []
    numerical_features_matrix = []
    # compute the value of n - dependent on the maximum of all the values below
    n = max(stock_long, stock_short, stock_signal, stock_n_trends, index_long, index_short, index_signal, index_n_trends)
    # for every value with at least n-1 preceding values
    for idx in range(n, len(date_pairs)):
        datum = []  # initialize the corresponding row of the numerical features matrix
        new_date_pairs.append(date_pairs[idx])  # add the new date and percentage change
        new_percentage_changes.append(percentage_changes[idx])

        # compute the n-previous percentage changes of the daily stock prices
        stock_n_prev_percentage_changes = percentage_changes[idx - stock_n_trends:idx]
        stock_n_prev_percentage_changes.reverse()

        # compute the n-previous trends of the daily stock prices
        stock_n_prev_trends = compute_trend(stock_n_prev_percentage_changes)

        # compute the stepwise positive trend occurrences in the daily stock prices
        stock_stepwise_num_pos = compute_stepwise_num_pos(stock_n_prev_trends)

        # compute the distance between this data point and the last time a positive
        # trend in the daily stock prices was seen
        stock_last_pos = list([get_last_pos(percentage_changes[:idx])])

        # compute the classified percentage changes in the daily stock prices
        stock_classified_changes = classify_magnitude(stock_n_prev_percentage_changes)

        # compute the n-gram vector for the daily stock prices
        stock_ngrams = get_ngrams(stock_ngram_size, stock_n_prev_trends)
        stock_ngram_vector = classify_ngrams(stock_ngrams)

        # compute the MACD technical indicator for the daily stock prices
        stock_long_ma = compute_mean(stock_prices[idx - stock_long:idx])
        stock_short_ma = compute_mean(stock_prices[idx - stock_short:idx])
        stock_signal_ma = compute_mean(stock_prices[idx - stock_signal:idx])
        stock_macd = stock_short_ma - stock_long_ma

        # compute the n-previous percentage changes of the daily snp500 index prices
        index_n_prev_percentage_changes = index_percentage_changes[idx - index_n_trends:idx]
        index_n_prev_percentage_changes.reverse()

        # compute the n-previous trends of the daily snp500 index prices
        index_n_prev_trends = compute_trend(index_n_prev_percentage_changes)

        # compute the stepwise positive trend occurrences in the daily snp500
        # index prices
        index_stepwise_num_pos = compute_stepwise_num_pos(index_n_prev_trends)

        # compute the distance between this data point and the last time a positive
        # trend in the daily snp500 index prices was seen
        index_last_pos = list([get_last_pos(index_percentage_changes[:idx])])

        # compute the classified percentage changes in the daily snp500 index prices
        index_classified_changes = classify_magnitude(index_n_prev_percentage_changes)

        # compute the n-gram vector for the daily snp500 index prices
        index_ngrams = get_ngrams(index_ngram_size, index_n_prev_trends)
        index_ngram_vector = classify_ngrams(index_ngrams)

        # compute the MACD technical indicator for the daily snp500 index prices
        index_long_ma = compute_mean(index_prices[idx - index_long:idx])
        index_short_ma = compute_mean(index_prices[idx - index_short:idx])
        index_signal_ma = compute_mean(index_prices[idx - index_signal:idx])
        index_macd = index_short_ma - index_long_ma

        # add the numerical features (including MACD) of the daily stock prices
        datum += stock_classified_changes
        datum += stock_n_prev_trends
        datum += stock_stepwise_num_pos
        datum += stock_last_pos
        datum += stock_ngram_vector

        datum += list([stock_long_ma])
        datum += list([stock_short_ma])
        datum += list([stock_macd])
        datum += list([stock_signal_ma])

        # add the numerical features (including MACD) of the daily snp500 index
        # prices
        #datum += index_classified_changes
        datum += index_n_prev_trends
        datum += index_stepwise_num_pos
        #datum += index_last_pos
        datum += index_ngram_vector

        datum += list([index_long_ma])
        datum += list([index_short_ma])
        datum += list([index_macd])
        datum += list([index_signal_ma])

        # add the row of numerical features to the matrix
        numerical_features_matrix.append(list(datum))
    # return the new date pairs, new stock price percentage changes and the numerical
    # features matrix
    return new_date_pairs, new_percentage_changes, numerical_features_matrix


