import pickle  # For loading a trained classifier
import nltk
import os


# Load functions for classifiers and datasets to save computation time
def load_classifier():
    with open('./saved/classifiers/naivebayes_scifi-fantasy_classifier.pickle',
              'rb') as f:
        classifier = pickle.load(f)
    f.closed
    return classifier


# Declare variables
categories = list()  # Categories (genres) to be classified
books_loaded = 10  # Number of training examples in each category
books = list()  # Tuples of book word lists and genre (book_words, genre)
excerpts = list()  # Tuples of word lists from book excerpts and their genre
all_word_freqs = list()  # Combined genre words and their frequencies


# Define extraneous  words with NLTK's default and custom book stop words
default_stopwords = set(nltk.corpus.stopwords.words('english'))
custom_stopwords = set(('sci-fi', 'fantasy', 'page', 'chapter', 'said'))

all_stopwords = default_stopwords | custom_stopwords

# Populate category list with the genres
for contents in os.listdir("./books/"):
    if str(contents) != "rename_and_shuffle_books.py":
        if str(contents) != "snippets":
            genre = str(contents)
            categories.append(genre)

# Create list of tuples of each book's tokenized words in a list and their
# corresponding genre. In the form [ ([book_words], genre) ] where [] = list
# and () = tuple
for genre in categories:
    directory = "./books/" + genre + "/txt/"
    books.extend(
                # Start at book #99 as previous 99 were used to train
                (list(nltk.word_tokenize(
                                        open(directory + genre +
                                             "_book_{}.txt".format(
                                                book_num + 100),
                                             'r', encoding='utf-8').read()
                                        )
                      ),
                 genre) for book_num in range(books_loaded))

# Create tuples for book excerpts and the genre they were taken from
for genre in categories:
    directory = "./book_excerpts/" + genre + "/"
    excerpts.extend(
                    # Extract 16 excerpts from each category for 32 total
                    (list(nltk.word_tokenize(
                                            open(directory + genre +
                                                 "_text_{}.txt".format(
                                                    excerpt_num),
                                                 'r').read()
                                            )
                          ),
                     genre) for excerpt_num in range(16))


# Extract features as top 2000 most frequently occurring words in books
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


# Separate test data into a list of tuples for classifier
test_set = [(book_features(book), genre) for (book, genre) in books]

# Load NaiveBayes Classifier
classifier = load_classifier()


# Performance Metric Measurements
positive = "fantasy"  # Assume fantasy to be positive
negative = "scifi"  # Assume Sci-fi to be negative

TP = 0  # True Positive
FP = 0  # False Positive
FN = 0  # False Negative
TN = 0  # True Negative
# Classify the test set and count the resulting true/false positives/negatives
for test_book in test_set:
    classified_genre = classifier.classify(test_book[0])
    correct_genre = test_book[1]
    # if true and a positive example increment true positive count
    if classified_genre == correct_genre and classified_genre == positive:
        TP += 1
    # if false and a positive example increment false positive count
    elif classified_genre != correct_genre and classified_genre == positive:
        FP += 1
    # if true and a negative example increment true negative count
    elif classified_genre == correct_genre and classified_genre == negative:
        TN += 1
    # if false and a negative example increment false negative count
    elif classified_genre != correct_genre and classified_genre == negative:
        FN += 1

sensitivity = round((TP / (TP + FN)) * 100, 2)
specificity = round((TN / (TN + FP)) * 100, 2)
accuracy = round(((TN + TP) / (TN + TP + FN + FP)) * 100, 2)
print("Classifier's performance metrics using the book test dataset: \n" +
      "Sensitivity: {:0.2f}% \nSpecificity: {:0.2f}% \nAccuracy: {:0.2f}%"
      .format(sensitivity, specificity, accuracy))


# Repeat  process for excerpt dataset
excerpt_set = [(book_features(xcerpt), genre) for (xcerpt, genre) in excerpts]

TP = 0  # True Positive
FP = 0  # False Positive
FN = 0  # False Negative
TN = 0  # True Negative
# Classify the test set and count the resulting true/false positives/negatives
for test_excerpt in excerpt_set:
    classified_genre = classifier.classify(test_excerpt[0])
    correct_genre = test_excerpt[1]
    # if true and a positive example increment true positive count
    if classified_genre == correct_genre and classified_genre == positive:
        TP += 1
    # if false and a positive example increment false positive count
    elif classified_genre != correct_genre and classified_genre == positive:
        FP += 1
    # if true and a negative example increment true negative count
    elif classified_genre == correct_genre and classified_genre == negative:
        TN += 1
    # if false and a negative example increment false negative count
    elif classified_genre != correct_genre and classified_genre == negative:
        FN += 1

sensitivity = round((TP / (TP + FN)) * 100, 2)
specificity = round((TN / (TN + FP)) * 100, 2)
accuracy = round(((TN + TP) / (TN + TP + FN + FP)) * 100, 2)
print("Classifier's performance using the book excerpt test dataset:\n" +
      "Sensitivity: {:0.2f}% \nSpecificity: {:0.2f}% \nAccuracy: {:0.2f}%"
      .format(sensitivity, specificity, accuracy))

num_of_features = len(classifier._feature_probdist.items())
print("The classifiers 10 most informative features (total of {} features): "
      .format(num_of_features))
classifier.show_most_informative_features(10)
