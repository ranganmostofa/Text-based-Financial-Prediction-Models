import process
import format_text
from time import time
from sklearn import svm
from sklearn import preprocessing
import numerical_features_extraction
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectPercentile, f_classif, chi2


WFC_CSV_FILENAME = "WFC Jan 02, 2015 - Jun 20, 2016.csv"  # set the WFC CSV filename

"""

Use any of the desired date ranges below:

WFC_CSV_FILENAME = "WFC Oct 01, 2014 - Jul 01, 2016.csv"
WFC_CSV_FILENAME = "WFC Oct 01, 2014 - Jun 21, 2016.csv"
WFC_CSV_FILENAME = "WFC Jan 02, 2015 - Jun 20, 2016.csv"
WFC_CSV_FILENAME = "WFC May 02, 2016 - Jun 20, 2016.csv"

NOTE: Date range must match with that for SNP

"""

SNP500_CSV_FILENAME = "SNP500 Jan 02, 2015 - Jun 20, 2016.csv"  # set the S&P 500 CSV filename

"""

Use any of the desired date ranges below:

SNP500_CSV_FILENAME = "WFC Oct 01, 2014 - Jul 01, 2016.csv"
SNP500_CSV_FILENAME = "WFC Oct 01, 2014 - Jun 21, 2016.csv"
SNP500_CSV_FILENAME = "WFC Jan 02, 2015 - Jun 20, 2016.csv"
SNP500_CSV_FILENAME = "WFC May 02, 2016 - Jun 20, 2016.csv"

NOTE: Date range must match with that for WFC

"""

STOCK_TICKER_SYMBOL = "WFC"  # set the stock ticker symbol
DATE_DELIMITER = " - "
DELIMITER = ","

K = 5  # set the number of folds
TRIALS = 1  # set the number of times the learning algorithm is run
TRAIN_PERCENTAGE = float((K - 1)) / K  # set the percentage of the dataset that is to be used for training

PERCENTILE = 1  # set the percentile of features used for training

# set the parameters for the numerical features extracted from the daily stock and snp500 prices

STOCK_LONG = 23
STOCK_SHORT = 12
STOCK_SIGNAL = 9

STOCK_N_TRENDS = 12
STOCK_NGRAM_SIZE = 2

INDEX_LONG = 23
INDEX_SHORT = 12
INDEX_SIGNAL = 9

INDEX_N_TRENDS = 11
INDEX_NGRAM_SIZE = 8

t_load_start = time()

wfc_data = process.load_csv(WFC_CSV_FILENAME)  # load the WFC CSV data file

snp500_data = process.load_csv(SNP500_CSV_FILENAME)  # load the S&P 500 CSV data file

dates = process.extract_column(wfc_data, "Date")  # get the dates from the csv
wfc_opening_prices = process.extract_column(wfc_data, "Open")  # get the opening prices
wfc_adj_closing_prices = process.extract_column(wfc_data, "Adj Close")  # get the adjusted closing prices for the stock

snp500_opening_prices = process.extract_column(snp500_data, "Open")  # get the opening prices for the index

date_pairs = process.create_pairs(dates)  # create pairs of dates
# compute percentage changes between consecutive opening prices of the stock
consecutive_pct_changes = process.compute_consecutive_pct_change(wfc_opening_prices)
# compute percentage changes between opening and closing prices of same day (stock)
daily_percentage_changes = process.compute_daily_pct_change(wfc_opening_prices[:len(wfc_opening_prices) - 1],
                                                            wfc_adj_closing_prices[:len(wfc_adj_closing_prices) - 1])
# compute percentage changes between consecutive opening prices of the index
snp500_changes = process.compute_consecutive_pct_change(snp500_opening_prices)

# load the news reports
WFC_NEWS_REPORTS = process.load_json("WFC NEWS REPORTS Oct 01, 2014 - Jul 01, 2016.json")

# load the completeness of the news
WFC_NEWS_STATUS = process.load_json("WFC NEWS STATUS (COMPLETE OR INCOMPLETE) Oct 01, 2014 - Jul 01, 2016.json")

t_load_end = time()

print "Time Taken to Load Data: " + str(t_load_end - t_load_start) + " s"
print

# for every date pair
for idx in range(len(date_pairs)):
    date_from, date_to = date_pairs[idx]
    date_key = date_from + DATE_DELIMITER + date_to
    # check if the news reports cover the time span between consecutive opening prices
    if WFC_NEWS_STATUS[date_key]:
        # if not, use the percentage change between opening and closing prices of the same day
        consecutive_pct_changes[idx] = daily_percentage_changes[idx]

# get the numerical features matrix and the shortened date pairs and percentage changes in daily stock prices
date_pairs, consecutive_pct_changes, numerical_features_matrix = numerical_features_extraction.numerical_features\
    (STOCK_LONG, STOCK_SHORT, STOCK_SIGNAL, STOCK_N_TRENDS, STOCK_NGRAM_SIZE,
     INDEX_LONG, INDEX_SHORT, INDEX_SIGNAL, INDEX_N_TRENDS, INDEX_NGRAM_SIZE,
     date_pairs, consecutive_pct_changes, wfc_opening_prices, snp500_changes, snp500_opening_prices)

# compute the trends in the daily stock prices
total_trends = process.normalize(consecutive_pct_changes)

# initialize variables to hold important information later
fold = 0
scores = []
weighted_scores = []
turn_scores = []
turn_weighted_scores = []
k_fold = KFold(n=len(total_trends), n_folds=5)

for train_indices, test_indices in k_fold:  # for every fold
    t0 = time()

    fold += 1

    print "Fold Number: " + str(fold)
    print

    # create the training date pairs, percentage changes, trends and numerical features matrix
    train_date_pairs = [date_pairs[i] for i in train_indices]
    train_pct_changes = [consecutive_pct_changes[i] for i in train_indices]
    train_trends = [total_trends[i] for i in train_indices]
    train_numerical_features_matrix = [numerical_features_matrix[i] for i in train_indices]

    # create the testing date pairs, percentage changes, trends and numerical features matrix
    test_date_pairs = [date_pairs[i] for i in test_indices]
    test_pct_changes = [consecutive_pct_changes[i] for i in test_indices]
    test_trends = [total_trends[i] for i in test_indices]
    test_numerical_features_matrix = [numerical_features_matrix[i] for i in test_indices]

    # build the train corpus
    train_corpus = process.build_corpus(train_date_pairs, WFC_NEWS_REPORTS)
    stemmed_train_corpus = format_text.stem_corpus(train_corpus)  # stem the train corpus

    # build the test corpus
    test_corpus = process.build_corpus(test_date_pairs, WFC_NEWS_REPORTS)
    stemmed_test_corpus = format_text.stem_corpus(test_corpus)  # stem the test corpus

    # instantiate the tfidf vectorizer object
    tfidf = TfidfVectorizer(min_df=2, max_df=0.80, ngram_range=(3, 4))

    # compute the textual features train matrix
    train_textual_features_matrix = tfidf.fit_transform(stemmed_train_corpus).toarray()

    # compute the textual features test matrix
    test_textual_features_matrix = tfidf.transform(stemmed_test_corpus).toarray()

    """

    Uncomment the Count Vectorizer and comment the TF-IDF Vectorizer if you wish to use Presence/Count Vectors

    # instantiate the count/presence vectorizer object
    count_vectorizer = CountVectorizer(binary=False, ngram_range=(3, 4))

    # compute the textual features train matrix
    train_textual_features_matrix = count_vectorizer.fit_transform(stemmed_train_corpus).toarray()

    # compute the textual features test matrix
    test_textual_features_matrix = count_vectorizer.transform(stemmed_test_corpus).toarray()

    NOTE: To use Presence Vectors, change the value of the parameter 'binary' to True when instantiating the
          Count Vectorizer object

    """

    # instantiate a Chi-Squared feature selector
    selector = SelectPercentile(chi2, percentile=1)

    selector.fit(train_textual_features_matrix, train_trends)

    # instantiate the numerical features matrix scaler - Standard Scaling is used here
    numerical_features_scaler = preprocessing.StandardScaler()

    # scale the numerical features matrix for training
    scaled_train_numerical_features_matrix = numerical_features_scaler.fit_transform(train_numerical_features_matrix)

    # scale the numerical features matrix for testing
    scaled_test_numerical_features_matrix = numerical_features_scaler.fit_transform(test_numerical_features_matrix)

    # create an instance of the model
    model = svm.SVC(kernel='linear', C=1, gamma='auto')

    """

    Use any of the desired models below:

    model = tree.DecisionTreeClassifier(min_samples_split=100)
    model = RandomForestClassifier(min_samples_split=100)
    model = GaussianNB()
    model = BernoulliNB()
    model = MultinomialNB()
    model = QuadraticDiscriminantAnalysis()
    model = GradientBoostingClassifier(max_depth=10, min_samples_leaf=6, min_samples_split=100)
    model = AdaBoostClassifier()
    model = svm.SVC(kernel='linear', C=1, gamma='auto')

    """

    # train the model
    model.fit(numerical_features_extraction.append_numerical_features
              (selector.transform(train_textual_features_matrix), scaled_train_numerical_features_matrix), train_trends)

    """

    Use either textual features (with or without feature selection) only, numerical features only or
    both types of features combined:

    model.fit(train_textual_features_matrix, train_trends)
    model.fit(selector.transform(train_textual_features_matrix), train_trends)
    model.fit(scaled_train_numerical_features_matrix, train_trends)
    model.fit(numerical_features_extraction.append_numerical_features
              (selector.transform(train_textual_features_matrix), scaled_train_numerical_features_matrix), train_trends)

    NOTE: For validity, you must use the same type or combination of features below for testing

    """

    # test the model
    predicted_trends = model.predict(numerical_features_extraction.append_numerical_features
                                     (selector.transform(test_textual_features_matrix),
                                      scaled_test_numerical_features_matrix))

    """

    Use either textual features (with or without feature selection) only, numerical features only or
    both types of features combined:

    #predicted_trends = model.predict(test_textual_features_matrix)
    #predicted_trends = model.predict(selector.transform(test_textual_features_matrix))
    #predicted_trends = model.predict(scaled_test_numerical_features_matrix)
    predicted_trends = model.predict(numerical_features_extraction.append_numerical_features
                                     (selector.transform(test_textual_features_matrix),
                                      scaled_test_numerical_features_matrix))

    NOTE: For validity, you must use the same type or combination of features above for training

    """

    print "Test Trends:", test_trends
    print
    print "Predicted Trends:", predicted_trends
    print

    # compute the accuracy of the model
    accuracy = accuracy_score(test_trends, predicted_trends)

    print "Simple Accuracy: " + str(accuracy * 100) + " %"
    print

    # initialize counter for weighted accuracy
    weighted_total = 0.00
    for idx in range(len(test_pct_changes)):  # for every datum in the test dataset
        if predicted_trends[idx] == test_trends[idx]:  # if the prediction is correct
            weighted_total += abs(test_pct_changes[idx])  # add the percentage change to the counter above
    # compute the weighted accuracy
    weighted_accuracy = weighted_total / sum([abs(number) for number in test_pct_changes])

    print "Weighted Accuracy: " + str(weighted_accuracy * 100) + " %"
    print

    # initialize variables for (simple and weighted) turn accuracy computation
    turn_total = 0
    turn_count = 0
    turn_weighted_total = 0.00
    turn_weighted_count = 0.00
    for idx in range(1, len(test_trends)):  # for every datum in the test dataset
        if test_trends[idx] != test_trends[idx - 1]:  # if this is a turning point
            turn_total += 1  # increase the counters appropriately
            turn_weighted_total += abs(test_pct_changes[idx])
            if predicted_trends[idx] == test_trends[idx]:  # if the prediction is correct
                turn_count += 1  # increase the counters appropriately
                turn_weighted_count += abs(test_pct_changes[idx])
    # compute the simple and weighted turn accuracies
    turn_accuracy = float(turn_count) / turn_total
    turn_weighted_accuracy = turn_weighted_count / turn_weighted_total

    print "Turn Accuracy: " + str(turn_accuracy * 100) + " %"
    print

    print "Turn Weighted Accuracy: " + str(turn_weighted_accuracy * 100) + " %"
    print

    # record all types of accuracies
    scores.append(accuracy)

    weighted_scores.append(weighted_accuracy)

    turn_scores.append(turn_accuracy)

    turn_weighted_scores.append(turn_weighted_accuracy)

    print "Time Taken: " + str(time() - t0) + " s"
    print


print "Cross-Validated Simple Accuracy: " + str(sum(scores) / len(scores) * 100) + " %"
print

print "Cross-Validated Weighted Accuracy: " + str(sum(weighted_scores) / len(weighted_scores) * 100) + " %"
print

print "Cross-Validated Turn Accuracy: " + str(sum(turn_scores) / len(turn_scores) * 100) + " %"
print

print "Cross-Validated Turn Weighted Accuracy: " + str(sum(turn_weighted_scores) /
                                                       len(turn_weighted_scores) * 100) + " %"
print


