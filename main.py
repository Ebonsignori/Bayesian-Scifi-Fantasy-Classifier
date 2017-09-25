import pickle  # For saving and loading a trained classifier
import nltk
import os
import random  # To randomize order of training examples

categories = list()  # Categories (genres) to be classified
num_of_books_in_cat = dict()  # Number of taining examples in each category
books = list()  # Tuples of book word lists and genre (book_words, genre)
all_word_freqs = list()  # Combined genre words and their frequencies


# Save functions for classifiers and data_sets to save computation time
def save_classifier(classifier):
    with open('./saved/classifiers/naivebayes_scifi-fantasy_classifier.pickle',
              'wb') as f:
        pickle.dump(classifier, f)
    f.close()


def save_data(data_set):
    with open('./saved/data_sets/scifi-fantasy_data.pickle', 'wb') as f:
        pickle.dump(data_set, f)
    f.close()

books_loaded = 5

# Define extrenuous words with nltk's default and custom book stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))
custom_stopwords = set(('sci-fi', 'fantasy', 'science fiction',
                        'page', 'chapter', 'said'))

all_stopwords = default_stopwords | custom_stopwords

# Populate cat list with categories and cat dict with their respective lengths
for contents in os.listdir("./Books/"):
    if str(contents) != "rename.py":
        genre = str(contents)
        categories.append(genre)
        num_of_books_in_cat[genre] = len(os.listdir("./Books/" +
                                         str(genre) + "/txt/"))

# Create list of all lists containing a book's words and corresponding label
for genre in categories:
    directory = "./Books/" + genre + "/txt/"
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

# Find 2000 most frequent words in the overall corpus of scifi + fantasy
for n in range(books_loaded*2):
    all_word_freqs.extend(
              nltk.FreqDist(word.lower() for word in books[n][0]
                            if not word.isnumeric() and  # Remove numbers
                            word.isalnum() and  # Remove punctuation
                            len(word) > 1 and  # Remove single characters
                            word not in all_stopwords  # Remove stopwords
                            )
                          )
word_features = list(all_word_freqs)[:2000]


# Define document features as words 'contained' in books
def book_features(book):
    book_words = set(book)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in book_words)
    return features

# Seperate data into training and test sets and save data for later
featuresets = [(book_features(book), genre) for (book, genre) in books]
train_set, test_set = featuresets[books_loaded*2:], featuresets[:books_loaded*2]
save_data(featuresets)

# Classify data using a NaiveBayes Classifier and save classifier for later
classifier = nltk.NaiveBayesClassifier.train(train_set)
save_classifier(classifier)

# Print accuracy of classifier with respect to the test set
print(nltk.classify.accuracy(classifier, test_set))
