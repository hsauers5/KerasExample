import csv
from Student import Student

splitter = 10806
with open('data/atriskyouth.csv') as csv_file:
    with open('data/training.csv', 'a+') as training:
        with open('data/testing.csv', 'a+') as testing:
            csv_reader = csv.reader(csv_file, delimiter=',')
            index = 0
            for row in csv_reader:
                index += 1
                if index <= splitter:
                    training.write(str(row).replace('[', '').replace(']', '').replace(' ', '').replace("'", '') + '\n')
                else:
                    testing.write(str(row).replace('[', '').replace(']', '').replace(' ', '').replace("'", '') + '\n')
