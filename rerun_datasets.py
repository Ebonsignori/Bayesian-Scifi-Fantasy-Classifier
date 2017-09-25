import pickle  # For loading a trained classifier
import nltk


# Load functions for classifiers and datasets to save computation time
def load_classifier():
    with open('./saved/classifiers/naivebayes_scifi-fantasy_classifier.pickle',
              'rb') as f:
        classifier = pickle.load(f)
    f.closed
    return classifier


def load_data():
    with open('./saved/datasets/scifi-fantasy_data.pickle', 'rb') as f:
        dataset = pickle.load(f)
    f.closed
    return dataset


featuresets = load_data()
train_set = featuresets[int(len(featuresets)*.2):]
test_set = featuresets[:int(len(featuresets)*.2)]

# Load NaiveBayes Classifier
classifier = load_classifier()

# Print accuracy of classifier with respect to the test set
print("Classifier accuracy = " +
      str(nltk.classify.accuracy(classifier, test_set)*100) + "%")
