import pickle  # For saving a trained classifier and datasets
import tkinter as tk
from tkinter import filedialog as fd  # For prompting user for book location
import nltk

print("Classify a new:\n[1].txt utf-8 book file \n[2] Manual text entry")

while True:
    option = input('Please enter 1 or 2 \n')
    try:
        option = int(option)
    except ValueError:
        print('Valid number, please')
        continue
    if option == 1 or option == 2:
        break
    else:
        print('Invalid Option')

if (option == 1):
    root = tk.Tk()
    root.withdraw()

    file_path = fd.askopenfilename()

    with open(file_path, 'r', encoding='utf-8') as f:
        book = nltk.word_tokenize(f.read())
else:
    book = input("Enter text to be classified: \n")
    book = nltk.word_tokenize(book)


# Load classifier
def load_classifier():
    with open('./saved/classifiers/naivebayes_scifi-fantasy_classifier.pickle',
              'rb') as f:
        classifier = pickle.load(f)
    f.closed
    return classifier


# Define extraneous words with NLTK's default and custom book stop words
default_stopwords = set(nltk.corpus.stopwords.words('english'))
custom_stopwords = set(('sci-fi', 'fantasy', 'page', 'chapter', 'said'))

all_stopwords = default_stopwords | custom_stopwords


# Extract features as top 2000 most occurring words in books
def book_features(book):
    # Find 2000 most frequent words in book
    most_frequent = nltk.FreqDist(word.lower() for word in book
                                  if not word.isnumeric() and  # Remove numbers
                                  word.isalnum() and  # Remove punctuation
                                  len(word) > 1 and  # Remove single characters
                                  word not in all_stopwords  # Remove stopwords
                                  )
    most_frequent = list(most_frequent)[:2000]  # Only top 2000

    # Define unique word occurrences in book
    book_words = set(book)
    features = {}

    # For each unique word in the book found in most frequent words in the book
    # add the 'word' to a dictionary with corresponding key, contains('word')
    for word in most_frequent:
        features['contains({})'.format(word)] = (word in book_words)
    return features


# Load Naive Bayes Classifier
classifier = load_classifier()
extracted_features = book_features(book)
genre = classifier.classify(extracted_features)
probability = classifier.prob_classify(extracted_features).prob(genre)

# Print accuracy of classifier with respect to the test set
print("\nThe classifier claims that your book genre is: " + genre)
print("{:.2f}% likely to be ".format(probability * 100) + genre)
