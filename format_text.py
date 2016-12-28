import re
import nltk
import nltk.tokenize as tk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


MIN_DF, MAX_DF = 2, 0.80  # set the parameters for the corpus-dependent stopwords extraction algorithm


def extract_sentences(text):
    """
    Given a text represented as a string, splits the string into the different
    sentences and returns a list of the sentences
    """
    # use regex to split the sentences
    sentence_list = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    final_sentence_list = []  # initialize a final sentence list
    for sentence in sentence_list:  # for every sentence in the list
        if sentence != "":  # if the sentence is not just a null string
            final_sentence_list.append(sentence)  # append to the final list
    return final_sentence_list  # return the final list of strings


def extract_words(text):
    """
    Given some text represented as a string, returns a list of all the words
    in the input text
    """
    word_list = text.split(" ")  # split the text using a space delimiter
    word_list = lowercase(word_list)  # convert all the words to lowercase
    # remove all leading and trailing non-alphanumeric characters from each word
    word_list = remove_non_alnums(word_list)
    word_list = remove_apostrophe_s(word_list)  # remove the apostrophe s from each word
    return word_list  # return the list of words


def lowercase(word_list):
    """
    Given a list of words as strings represented as strings, returns a list of
    the same words in lowercase
    """
    new_word_list = []  # initialize an empty list of new words
    for word in word_list:  # for every word in the word list
        # append the lowercase version of the word to the new word list
        new_word_list.append(word.lower())
    return new_word_list  # return the new word list


def remove_non_alnums(word_list):
    """
    Given a list of words represented as strings, returns a list of the same words
    once all leading and trailing non-alphanumeric characters have been removed from
    each word
    """
    new_word_list = []  # initialize an empty list of new words
    for word in word_list:  # for every word in the word list
        start_idx = 0  # initialize the start index of the new word
        for character in word:  # for every character in the word
            if character.isalnum():  # if the character is alphanumeric
                break  # stop looping since the start index has been found
            start_idx += 1  # otherwise increase start_idx by 1
        # if by any chance the start index is longer than the word itself
        if start_idx >= len(word):
            # the word has no alphanumeric content and so has to be removed
            continue  # hence, skip an iteration
        end_idx = start_idx  # initialize the end index of the new word
        # for every character in the rest of the word
        for character in word[start_idx:]:
            if not character.isalnum():  # if the character is not alphanumeric
                break  # stop looping since the end index has been found
            end_idx += 1  # otherwise increase the end index by 1
        # however if the last character is alphanumeric
        if word[-1].isalnum():
            # the new word is from the start index onwards
            new_word = word[start_idx:]  # this takes care of words with apostrophes
        else:  # otherwise
            # use splicing to remove the non-alphanumeric content
            new_word = word[start_idx:end_idx]
        new_word_list.append(new_word)  # append the new word to the new word list
    return new_word_list  # return the new word list


def remove_apostrophe_s(word_list):
    """
    Given a list of words represented as strings, returns a list of the same words
    without any apostrophe s-es
    """
    apostrophe_s = "'s"
    new_word_list = []  # initialize an empty list of new words
    for word in word_list:  # for every word in the word list
        if word.endswith(apostrophe_s):  # if it ends with an apostrophe s
            new_word = word[:len(word) - 2]  # extract just the root of the word
            new_word_list.append(new_word)  # append the root to the new list
        else:  # otherwise if the word has no apostrophe s-es
            new_word_list.append(word)  # just append the original word
    return new_word_list  # return the new word list


def join_words(word_list, delimiter):
    """
    Given a list of words represented as strings, returns the sentence that is
    formed when the words are delimited by the delimiter
    """
    sentence = ""  # initialize a null string to hold the sentence
    for word in word_list:  # for every word in the string
        sentence += word + delimiter  # form the string
    # return the sentence except the last delimiter
    return sentence[:len(sentence) - 1]


def stem_words(word_list):
    """
    Given a list of words (strings), creates a list of the same words reduced
    to their base or root form and returns this list of stemmed words
    """
    stemmed_word_list = []  # initialize an empty word list
    stemmer = nltk.stem.porter.PorterStemmer()  # create a stemmer object
    for word in word_list:  # for every pair in the list
        # stem the word and add the base form to the new list along with the tag
        stemmed_word_list.append(stemmer.stem(word))
    return stemmed_word_list  # return the list of stemmed words and tags


def stem_corpus(corpus):
    """
    Given a corpus represented by a list of documents, returns a new list of
    documents that have been stemmed
    """
    new_corpus = []  # initialize empty list of stemmed documents
    for document in corpus:  # for every document in the input corpus
        # stem the document and add it to the list above
        new_corpus.append(" ".join(stem_words(tk.word_tokenize(document))))
    return new_corpus  # return the stemmed list of documents, i.e. stemmed corpus


def filter_stopwords(word_list, linguistic_stopwords):
    """
    Given a list of words (strings), identifies the stopwords, filters them out
    and returns a new list of words with no stopwords
    """
    processed_word_list = []  # initialize a new word list
    for word in word_list:  # for every word in the word list
        if word not in linguistic_stopwords:  # if the word is not a stop word
            processed_word_list.append(word)  # append it to the new list
    return processed_word_list  # return the new list


def filter_corpus_stopwords(corpus, stopwords):
    """
    Given a corpus represented as a list of documents and a list of stopwords
    (strings), returns a corpus or list of documents with the stopwords filtered off
    """
    new_corpus = []  # initialize an empty list of documents
    for document in corpus:  # for every document in the input corpus
        word_list = tk.word_tokenize(document)  # get a list of tokens in the document
        # filter off the stopwords in the list of tokens
        filtered_word_list = filter_stopwords(word_list, stopwords)
        # add the filtered document to the list of documents or corpus initialized above
        new_corpus.append(" ".join(filtered_word_list))
    return new_corpus  # return the filtered corpus


def get_nltk_stopwords():
    """
    Returns the stopwords from the nltk corpus
    """
    return stopwords.words("english")  # return the english stopwords


def get_tfidf_stopwords(corpus):
    """
    Given a corpus of documents represented as a list of strings, returns the stopwords
    identified by a tf-idf vectorizer
    """
    tfidf_vec = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF)  # create the vectorizer object
    tfidf_vec.fit(corpus)  # fit the vectorizer to the corpus
    return list(tfidf_vec.stop_words_)  # return the set of stopwords


def get_verbs(corpus):
    """
    Given a corpus represented as a list of documents (strings), returns a new corpus
    or list of documents where each documents have been reduced to just the verbs
    (delimited by a space) contained in the original document
    """
    # initialize the tuple of tags related to verbs
    verb_tags = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
    verbed_corpus = []  # initialize a new empty corpus
    for document in corpus:  # for every document in the input corpus
        verb_list = []  # initialize an empty list of verbs
        sentence_list = extract_sentences(document)  # extract the sentences in the document
        for sentence in sentence_list:  # for every sentence in the document
            # use the built-in POS tagger to tag the words with their respective part of speech
            tagged = nltk.pos_tag(extract_words(sentence))
            for word, pos_tag in tagged:  # for every word and POS tag pair
                if pos_tag in verb_tags:  # if the tag is related to verbs
                    # the word is a verb and so add it to the list of verbs initialized above
                    verb_list.append(word)
            # concatenate the verbs delimited by a space
            verbed_corpus.append(" ".join(verb_list))
    return verbed_corpus  # return the corpus consisting of documents containing only verbs


