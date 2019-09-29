import csv

import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from Student import Student

data_x = []
data_y = []
with open('data/training.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        # clean data
        for i in range(0, len(row)):
            if row[i] == '':
                row[i] = '0'
        this_student = Student(row)
        data_x.append(this_student.get_core_no_opiates())
        data_y.append(this_student.opiate_use)

data_length = len(data_x[0])
data_x = numpy.array(data_x)
data_y = numpy.array(data_y)

model = Sequential()
model.add(Dense(16, input_dim=data_length, activation='sigmoid'))  # sigmoid/relu12/tanh1, 94.95, 41/50
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data_x, data_y, validation_split=0.33, epochs=500, batch_size=20)

scores = model.evaluate(data_x, data_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# model.save_weights("model.h5")

save = input("Save file? y/n: ")
if save == 'y':
    model.save_weights("splitmodel.h5")
