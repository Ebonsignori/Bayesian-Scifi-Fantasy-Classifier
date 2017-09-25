import pickle  # For saving a trained classifier and datasets
import nltk
import sys

file_path = sys.argv[1]

with open(file_path, 'r', encoding='utf-8') as f:
    book = nltk.word_tokenize(f.read())


# Load classifier
def load_classifier():
    with open('./saved/classifiers/naivebayes_scifi-fantasy_classifier.pickle',
              'rb') as f:
        classifier = pickle.load(f)
    f.closed
    return classifier


# Define extrenuous words with nltk's default and custom book stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))
custom_stopwords = set(('sci-fi', 'fantasy', 'page', 'chapter', 'said'))

all_stopwords = default_stopwords | custom_stopwords

# Find 2000 most frequent words in the provided book
word_freqs = nltk.FreqDist(word.lower() for word in book
                           if not word.isnumeric() and  # Remove numbers
                           word.isalnum() and  # Remove punctuation
                           len(word) > 1 and  # Remove single characters
                           word not in all_stopwords  # Remove stopwords
                           )
word_features = list(word_freqs)[:2000]


# Define document features as words 'contained' in books
def book_features(book):
    book_words = set(book)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in book_words)
    return features

# Load NaiveBayes Classifier
classifier = load_classifier()

# Print accuracy of classifier with respect to the test set
print(classifier.classify(book_features(book)))
