import os
import random

categories = list()

for contents in os.listdir("./"):
    if str(contents) != "rename_and_shuffle_books.py":
        if str(contents) != "snippets":
            categories.append(str(contents))

for cat in categories:
    book_num = 0
    current_dir = "./" + str(cat) + "/txt/"
    books_in_dir = os.listdir(current_dir)
    random.shuffle(books_in_dir)
    for book in books_in_dir:
        os.rename(current_dir + book,
                  current_dir + str(cat) + "_book_" + str(book_num) +
                  ".txt")
        book_num += 1
