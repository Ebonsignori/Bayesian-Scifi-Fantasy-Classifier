# Sci-Fi/Fantasy Genre Classifier using Naive Bayes
Programs require: 
* Python 3 or higher (Source written and tested in 3.6)
* [pickle - Python Object Serialization](https://docs.python.org/3/library/pickle.html)
* [nltk - Natural Language Toolkit](http://www.nltk.org/install.html)

Programs allow to do the following:
1. Train a Bayesian classifier with fantasy and sci-fi books in .txt (utf-8) format
2. Calculate the accuracy (%) of said classifier using ROC methods
3. Attempt to classify a new book in .txt (utf-8) format as either sci-fi or fantasy

All three can be done in any order since the classifier has already been 
trained and stored in a binary file using pickle. The file can be found 
in *./saved/classifiers*. 


### (1) Training
To retrain the classifier you may rerun:

`python inital_train.py`

Be aware that it will take apx. 4 minutes to process the 200 training examples. 

### (2) Accuracy
To view the classifier's accuracy:

`python performance_test.py`

Include the *saved* folder in the same directory as *performance_test.py*. 

### (3) New Book Classification
To classify a new book you can pass a book already in the repository or your
own if you first convert it to a .txt format with utf-8 encoding.

```
python classify_new.py ./Books/scifi/txt/scifi_book_101.txt
>>> The classifier claims that your book genre is: scifi
```

or for you own book

```
python classify_new.py ./name_of_your_book.txt
>>> The classifier claims that your book genre is: fantasy
```

# Systems Diagram
![Alt text](./documentation/P1_diagram_png.png?raw=true "Systems Diagram")

### Viewing Fun and Useful Classification Information
To see what the most informative features that the classifier considers during
classification, run either (2) or (3) with the '-i' flag, but don't retrain the
classifier (1). 

Once the program completes execution, enter *classifier.show_most_informative_features()*
into the interactive python shell.
```
python -i rerun_datasets.py
>>> classifier.show_most_informative_features(20)
```
![Alt text](./documentation/20_most_informative.PNG?raw=true "20 Most Informative")