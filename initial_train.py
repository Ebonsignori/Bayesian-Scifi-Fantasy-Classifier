import pickle  # For saving a trained classifier and datasets
import nltk
import os
import random  # To randomize order of training examples
import time  # For timing execution of program


# Time program execution
time_taken = time.time()

# Declare variables
categories = list()  # Categories (genres) to be classified
books_loaded = 100  # Number of taining examples in each category
books = list()  # Tuples of book word lists and genre (book_words, genre)
all_word_freqs = list()  # Combined genre words and their frequencies


# Save functions for classifiers and datasets to save computation time
def save_classifier(classifier):
    with open('./saved/classifiers/naivebayes_scifi-fantasy_classifier.pickle',
              'wb') as f:
        pickle.dump(classifier, f)
    f.close()


def save_data(dataset):
    with open('./saved/datasets/scifi-fantasy_data.pickle', 'wb') as f:
        pickle.dump(dataset, f)
    f.close()

# Define extrenuous words with nltk's default and custom book stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))
custom_stopwords = set(('sci-fi', 'fantasy', 'page', 'chapter', 'said'))

all_stopwords = default_stopwords | custom_stopwords

# Populate cat list with categories and cat dict with their respective lengths
for contents in os.listdir("./books/"):
    if str(contents) != "rename_and_shuffle_books.py":
        if str(contents) != "snippets":
            genre = str(contents)
            categories.append(genre)

# Create list of all lists containing a book's words and corresponding label
for genre in categories:
    directory = "./books/" + genre + "/txt/"
    books.extend(
                (list(nltk.word_tokenize(
                                        open(directory + genre +
                                             "_book_{}.txt".format(book_num),
                                             'r', encoding='utf-8').read()
                                        )
                      ),
                 genre) for book_num in range(books_loaded))

# Randomize order of training examples and their genres
random.shuffle(books)


# Extract features as top 2000 most occuring words in books
def book_features(book):
    # Find 2000 most frequent words in book
    most_frequent = nltk.FreqDist(word.lower() for word in book
                                  if not word.isnumeric() and  # Remove numbers
                                  word.isalnum() and  # Remove punctuation
                                  len(word) > 1 and  # Remove single characters
                                  word not in all_stopwords  # Remove stopwords
                                  )
    most_frequent = list(most_frequent)[:2000]  # Only top 2000

    # Define unique word occurances in book
    book_words = set(book)
    features = {}

    # For each unique word in the book found in most frequent words in the book
    # add the 'word' to a dictionary with corresponding key, contains('word')
    for word in most_frequent:
        features['contains({})'.format(word)] = (word in book_words)
    return features

# Seperate data into training and test sets and save data for later
train_set = [(book_features(book), genre) for (book, genre) in books]
save_data(train_set)

# Classify data using a NaiveBayes Classifier and save classifier for later
classifier = nltk.NaiveBayesClassifier.train(train_set)
save_classifier(classifier)

# Print time taken for program execution
print("Time taken for " + str(books_loaded) + " books from each genre to train \
the Bayesian classifier: \n" + str(time.time() - time_taken) + " seconds")
