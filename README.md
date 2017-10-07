# Sci-Fi/Fantasy Genre Classifier using Naive Bayes
Programs require: 
* Python 3 or higher (Source written and tested in 3.6)
* [pickle - Python Object Serialization](https://docs.python.org/3/library/pickle.html)
* [nltk - Natural Language Toolkit](http://www.nltk.org/install.html)

Included programs can do the following:
1. Train a Bayesian classifier with fantasy and sci-fi books in .txt (utf-8) format
2. Calculate the accuracy (%) of said classifier using ROC methods and display the most informative features
3. Load a book in .txt (utf-8) format or accept manual text entry and classify your option as either sci-fi or fantasy content 

All three can be done in any order since the classifier has already been 
trained and stored in a binary file using pickle. The file can be found 
in *./saved/classifiers/*. 


### (1) Training
To retrain the classifier you may rerun:

`python inital_train.py`

Be aware that it will take apx. 4 minutes (depending on your system) to process the 200 training examples. 

### (2) Accuracy
To view the classifier's accuracy and top 10 most informative features:

`python performance_test.py`

Make sure to include the *saved* folder in the same directory as *performance_test.py*. 

### (3) New Classification
To classify a new book or text you are given the following option at runtime:
```
python classify_new.py
Classify a new:
[1].txt utf-8 book file
[2] Manual text entry
Please enter 1 or 2
>>> 
```

If you choose option [1], then you can either select a book provided in the repository or a book located elsewhere on your filesystem. If you use your own book, make sure to convert it to a .txt file in utf-8 encoding.


If you choose option [2], then manually enter the text you want to have classified:
```
python classify_new.py
Classify a new:
[1].txt utf-8 book file
[2] Manual text entry
Please enter 1 or 2
2
Enter text to be classified:
After the robots mined all of earth’s resources, we entered our spaceships and headed to the stars

The classifier claims that your book genre is: scifi
99.65% likely to be scifi
```

# Systems Diagram
![Alt text](./documentation/P1_diagram_png.png?raw=true "Systems Diagram")

# Performance Analysis
The classifier isn’t perfect, but it is almost perfect when functioning within its domain. I conjecture that the classifier would yield more than a 95% accuracy if classifying the domain of all science fiction and fantasy books. 
To defend this conjecture, let us assume that since human beings are the writers of sci-fi and fantasy books, human beings can read and correctly identify any book into one of the two genres. Now let us consider the more specific subset of single sentences from a sci-fi or fantasy book. I’ve compiled a list of 15 sentences that can be classified as “likely sci-fi”, “likely fantasy”, or “ambiguous”.  Consider just three of them and guess each of their classifications:
1.  “Pulling his sword out of its sheath, the knight lunged forward and decapitated the dragon.”
2.  “After the robots had mined all of earth’s resources, we took our ships to other stars”  
3.  “Just consider the data available to us and you’d know that dragons and magic aren’t real.” 

If we classify these sentences using classify_new.py’s manual text entry option, we get the following:
![Alt text](./documentation/manual_sentence_1.png?raw=true "Sentence 1 is 99.92% likely to be fantasy.")
![Alt text](./documentation/manual_sentence_2.png?raw=true "Sentence 2 is 98.45% likely to be scifi.")
![Alt text](./documentation/manual_sentence_3.png?raw=true "Sentence 3 is 50.80% likely to be scifi.")

Note that the last sentence, being intentionally ambiguous, had an accuracy of 50%. Apart from sentence 3, It’s likely that the classifier’s genres matched that of your own guesses. If we accept that the classifier can accurately classify a subset of a book, then given the nature of Bayesian classification, the classification of an entire book follows. Thus, using heuristic methods, we prove our conjecture that our classifier is incredibly accurate in its given domain.

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