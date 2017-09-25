import os

categories = list()

for contents in os.listdir("./"):
    if str(contents) != "rename.py":
        categories.append(str(contents))

for cat in categories:
    book_num = 0
    current_dir = "./" + str(cat) + "/txt/"
    for book in os.listdir(current_dir):
        os.rename(current_dir + book,
                  current_dir + str(cat) + "_book_" + str(book_num) + ".txt")
        book_num += 1
