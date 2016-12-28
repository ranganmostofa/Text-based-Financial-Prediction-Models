import csv
import json


def dict_to_vars(a_dict):
    """
    Given a dictionary mapping the independent variable to the dependent
    variable, separates the two into two different lists and returns them
    """
    x = []  # initialize lists
    y = []
    for key in a_dict.keys():  # for every value of the independent variable
        for value in a_dict[key]:  # for every value of the dependent variable
            x.append(list([key]))  # append the independent variable
            y.append(value)  # append the dependent variable
    return x, y  # return the two lists as an unpacked pair


def compute_consecutive_pct_change(data):
    """
    Given a data set represented as a list of values, returns a list of percentage
    changes between values with consecutive indices
    """
    pct_change = []  # initialize an empty list of percentage changes
    for idx in range(1, len(data)):  # for every value
        # calculate the percentage change with the next value and append it
        pct_change.append((data[idx] - data[idx - 1]) / data[idx - 1] * 100)
    return pct_change  # return the percentage change list


def compute_daily_pct_change(opening, closing):
    """
    Given the day-to-day opening and closing prices, represented as two lists of floats
    computes  and returns the daily percentage change in prices
    """
    pct_changes = []  # initialize an empty list of percentage changes
    for idx in range(len(opening)):  # for every pair of opening and closing prices
        pct_change = float(closing[idx] - opening[idx]) / opening[idx] * 100  # calculate the percentage change
        pct_changes.append(pct_change)  # append the percentage change to the list
    return pct_changes  # return the list of percentage changes


def compute_trend(data):
    """
    Given a list of daily stock prices, returns a list of trinary integers:
    * -1 when the price has fallen compared to the preceding price
    * 0 when both prices are equal
    * 1 when the price has risen compared to the preceding price
    """
    trends = []  # initialize lists
    for i in range(1, len(data)):  # for every datum in the input list
        # if the current datum (referenced by the index) is smaller than the preceding value
        if data[i] < data[i-1]:
            trends.append(-1)  # add the integer -1 to the list
        elif data[i] == data[i-1]:  # if both are equal
            trends.append(0)  # add the integer 0 to the list
        else:  # otherwise
            trends.append(1)  # add the integer 1 to the list
    return trends  # return the list of trends


def create_pairs(data):
    """
    Given a data set represented as a list of values, returns a list of pairs of
    consecutive data points in the input list
    """
    pairs = []  # initialize an empty list of pairs
    for idx in range(len(data) - 1):  # for every value
        # append the pair consisting of that value and the value after
        pairs.append(tuple((data[idx], data[idx + 1])))
    return pairs  # return the pairs list


def map_data(keys, values):
    """
    Given a list of keys and a list of values, creates and returns a dictionary
    """
    mapped_data = {}  # initialize the empty dictionary
    if len(keys) == len(values):  # if the lists are compatible, i.e. of equal length
        for idx in range(len(keys)):  # for every key
            mapped_data[keys[idx]] = values[idx]  # map it to its value
    return mapped_data  # return the mapped data


def build_corpus(date_pairs, news_reports):
    """
    Given a list of pairs of the form (date_from, date_to) and the news reports
    (call this one document) released from a start date to an end date inclusive,
    returns a list of the documents
    """
    corpus = []  # initialize the list of documents
    for date_pair in date_pairs:  # for every date pair in the input list of date pairs
        date_from, date_to = date_pair  # use unpacking to access the start date and end date
        date_key = date_from + " - " + date_to  # create the date key using a ' - ' as a delimiter
        corpus.append(news_reports[date_key])  # append this to the initialized list above
    return corpus  # return the list of documents/corpus


def compute_mse(actual_data, predicted_data):
    """
    Given an array of actual values and a corresponding array of values
    predicted by a model, returns the mean-squared error of the model fitted
    by that particular sample
    """
    # only run if the two data sets are compatible, i.e. of equal length
    if len(actual_data) == len(predicted_data):
        total_error = 0  # initialize the total error to zero
        for idx in range(len(actual_data)):  # for every data point
            # add the squared error to the total error
            total_error += (actual_data[idx] - predicted_data[idx]) ** 2
        # return the mean of the squared error
        return float(total_error) / len(actual_data)


def compute_mape(actual_data, predicted_data):
    """
    Given an array of actual values and a corresponding array of values
    predicted by a model, returns the mean absolute percentage error of the
    model fitted by that particular sample
    """
    # only run if the two data sets are compatible, i.e. of equal length
    if len(actual_data) == len(predicted_data):
        total_pct_error = 0  # initialize the total percentage error to zero
        for idx in range(len(actual_data)):  # for every data point
            # add the absolute percentage error to the total percentage error
            total_pct_error += (abs(actual_data[idx] - predicted_data[idx])) / float(actual_data[idx])
        # return the mean of the absolute percentage error
        return float(total_pct_error) / len(actual_data)


def compute_trend_accuracy(actual, predicted):
    """
    Given a list of actual values and a list of predicted values, returns the
    accuracy of the predictions
    """
    correct = 0  # initialize a running counter of correct predictions
    # if the two lists are compatible for comparison, i.e. if they are of the same length
    if len(actual) == len(predicted):
        # for every data point
        for idx in range(len(actual)):
            if actual[idx] == predicted[idx]:  # if the prediction matches the actual value
                correct += 1  # increase the "correct" counter by 1
        # return the accuracy as a decimal between 0 and 1 inclusive
        return float(correct) / len(actual)


def normalize(data):
    """
    Given a data set, maps all negative values to -1, all positive values
    to +1 and zero to 0
    """
    normalized_data = []  # initialize an empty list of new data
    for datum in data:  # for every datum in the data set
        if datum != 0.00:  # if the datum is not 0 (to avoid division by zero)
            # use the mapping formula to map negative values to -1 and positive value to +1
            normalized_data.append(int(datum / abs(datum)))
        else:  # if the datum is zero
            normalized_data.append(int(datum))  # append the zero
    return normalized_data  # return the new normalized data


def load_csv(csv_filename):
    """
    Given a csv file name, opens it and stores all the values in a matrix
    """
    data_matrix = []  # initialize the matrix
    with open(csv_filename, "rb") as csv_file:  # open the file
        csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")  # read the file
        for row in csv_reader:  # for every row, i.e. data field
            data_matrix.append(list(row))  # append it to the matrix
    return data_matrix  # return the data matrix


def save_json(obj, filename):
    """
    Given an object and a filename, creates a .json file with the provided
    filename, opens the json file and stores the Python object in the json file
    """
    with open(filename, "wb") as sc:  # create and open the json file
        json.dump(obj, sc)  # store the object
    sc.close()  # close the json file


def load_json(filename):
    """
    Given a json filename, opens the file, loads the object stored in the json
    file and returns this object
    """
    with open(filename, "rb") as fp:  # open the json file with the provided filename
        obj = json.load(fp)  # load the object stored in the json file
    fp.close()  # close the json file
    return obj  # return the object


def remove_heading(data_matrix):
    """
    Given a matrix of data, removes the heading, i.e. the first row of the
    matrix and returns the resulting matrix
    """
    return list(data_matrix[1:])  # use splicing to remove the heading


def extract_column(data_matrix, row_name):
    """
    Given a data matrix (a list of lists) with a headings column of strings
    and the heading of the desired column of data (a string), returns the column
    of data
    """
    data_matrix_copy = list(data_matrix)  # avoid mutation of input arguments
    labels = data_matrix_copy.pop(0)  # chop out the headings column
    col_idx = labels.index(row_name)  # get the index of the desired column

    column = []  # initialize an empty column (list) of data
    for row in data_matrix_copy:  # for every row in the data matrix
        # find the data value in the position under the desired column and append it to the
        # growing column of data
        if row_name == "Date":  # if the row heading says "Date" do not use type casting
            column.append(row[col_idx])
        else:  # otherwise do
            column.append(float(row[col_idx]))
    return column  # return the final column of data


def store_score_chart(headings, score_chart, csv_filename, delimiter):
    """
    Given a list of headings, a score chart and a csv file name, creates and stores
    the score chart in the csv file
    """
    with open(csv_filename, "w") as csv_file:  # open / create the csv file
        # create a csv writer object to write the data into the file using the delimiter ","
        score_writer = csv.writer(csv_file, delimiter=delimiter, quotechar="|", quoting=csv.QUOTE_MINIMAL)
        # write the heading
        score_writer.writerow(headings)
        for word in score_chart.keys():  # for every word
            # for frequency and the associated scores
            if delimiter not in word:
                for frequency, scores in score_chart[word].items():
                    # for every score
                    for score in scores:
                        # store the score in the order: word, frequency, score
                        try:
                            score_writer.writerow(list([word, str(frequency), str(score)]))
                        except Exception as exception_message:
                            print(exception_message)


def load_score_chart(csv_filename):
    """
    Given a csv file name containing a score chart, loads it in as a dictionary and
    returns the score chart
    """
    score_chart = {}  # initialize the score chart
    data_matrix = load_csv(csv_filename)  # load the score chart as a data_matrix
    data_matrix = remove_heading(data_matrix)  # remove the headings row
    for row in data_matrix:  # for every row
        word, frequency, score = row  # get the word, frequency and score
        if score_chart.has_key(word):  # if the score chart already has the word as a key
            # if the score chart already has the frequency as a key of the word
            if score_chart[word].has_key(int(frequency)):
                # append the score to the list of scores
                score_chart[word][int(frequency)].append(float(score))
            else:  # if the frequency is not a key of the word
                # add the frequency and the score associated with it
                score_chart[word][int(frequency)] = list([float(score)])
        else:  # if the word is not a key of the score chart
            # add the word as a key and include the inner dictionary mapping the frequency
            # to the score
            score_chart[word] = {int(frequency): list([float(score)])}
    return score_chart  # return the score chart


def store_bigram_score_chart(headings, score_chart, csv_filename, delimiter):
    """
    Given a list of headings, a score chart and a csv file name, creates and stores
    the score chart in the csv file
    """
    with open(csv_filename, "wb") as csv_file:  # open / create the csv file
        # create a csv writer object to write the data into the file using the delimiter ","
        score_writer = csv.writer(csv_file, delimiter=delimiter, quotechar="|", quoting=csv.QUOTE_MINIMAL)
        # write the heading
        score_writer.writerow(headings)
        for word in score_chart.keys():  # for every word
            # for frequency and the associated scores
            if delimiter not in word:
                for neighboring_word in score_chart[word].keys():  # for every neighboring word
                    if delimiter not in neighboring_word:
                        for frequency, scores in score_chart[word][neighboring_word].items():
                            # for every frequency and score pair
                            for score in scores:
                                # store the score in the order: word, neighboring word, frequency, score
                                try:
                                    score_writer.writerow(list([word, neighboring_word, str(frequency), str(score)]))
                                except Exception as exception_message:
                                    print exception_message


def get_transpose(matrix):
    """
    Given a matrix represented as a list of lists, returns the transpose of the matrix
    """
    transpose = []  # initialize an empty matrix
    for col_idx in range(len(matrix[0])):  # for every column
        row = list([])  # initialize the corresponding row of the transpose
        for row_idx in range(len(matrix)):  # for every row
            row.append(matrix[row_idx][col_idx])  # add the corresponding element
        transpose.append(list(row))  # append the row to the transpose
    return transpose  # return the transpose


