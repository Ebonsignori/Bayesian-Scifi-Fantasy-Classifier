import pickle  # For saving and loading a trained classifier
import nltk


# Save functions for classifiers and data_sets to save computation time
def load_classifier():
    with open('naivebayes_scifi-fantasy_classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)
    f.closed
    return classifier


def load_data():
    with open('./saved/data_sets/scifi-fantasy_data.pickle', 'rb') as f:
        data_set = pickle.load(f)
    f.closed
    return data_set


featuresets = load_data()
train_set = featuresets[len(featuresets):]
test_set = featuresets[:len(featuresets)]

# Load NaiveBayes Classifier
classifier = load_classifier()

# Print accuracy of classifier with respect to the test set
print(nltk.classify.accuracy(classifier, test_set))
