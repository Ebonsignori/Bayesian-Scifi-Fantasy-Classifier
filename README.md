# Sci-Fi and Fantasy Genre Classifier using Naive Bayesian Algorithms
Use initial_training.py to train 200 book examples (100 sci-fi and 100 fantasy)
then pass a ".txt" file as an argument for classify_new.py

## Example
*python classify_new.py ./Books/scifi/txt/scifi_book_101.txt*
*>>> The classifier claims that your book genre is: scifi*

### Fun Information
To see what the classifier considers for classification, run either the intial_training.py
or rerun_datasets.py with the -i flag and then pass the method "show_most_informative_features()"
to the variable "classifier"

## Example
*python -i rerun_datasets.py*
*>>> classifier.show_most_informative_features(20)*

Lists the top 20 features in a nice table:
![Alt text](./imgs/20_most_informative.PNG?raw=true "20 Most Informative")