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
        data_y.append(this_student.ever_hard_drugs)

data_length = len(data_x[0])
data_x = numpy.array(data_x)
data_y = numpy.array(data_y)

model = Sequential()
model.add(Dense(150, input_dim=data_length, activation='tanh'))  # sigmoid/relu12/tanh1, 94.8, 53/56
model.add(Dense(72, activation='relu'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# model.load_weights('models/tanh150relu72tanh64relu32tanh1-27_29.h5')

model.fit(data_x, data_y, validation_split=0.33, epochs=100, batch_size=100)

scores = model.evaluate(data_x, data_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate metrics
data_x = []
data_y = []
with open('data/testing.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        # clean data
        for i in range(0, len(row)):
            if row[i] == '':
                row[i] = '0'
        this_student = Student(row)
        data_x.append(this_student.get_core_no_opiates())
        data_y.append(this_student.ever_hard_drugs)

data_length = len(data_x[0])
data_x = numpy.array(data_x)
data_y = numpy.array(data_y)

predictions = model.predict_classes(data_x)
false_positives = 0
false_negatives = 0
total_users = 0
users_caught = 0
for i in range(0, len(predictions)):
    if data_y[i] == 1 and predictions[i] == 0:
        false_negatives += 1
    elif data_y[i] == 0 and predictions[i] == 1:
        false_positives += 1
    if data_y[i] == 1:
        total_users += 1
        if predictions[i] == 1:
            users_caught += 1

    # print(str(i) + ' | ' + str(data_y[i]) + " vs predicted: " + str(predictions[i]))

print("False positives: " + str(false_positives))
print("False negatives: " + str(false_negatives))
print("Users caught: " + str(users_caught))
print("Total users: " + str(total_users))

save = input("Save file? y/n: ")
if save == 'y':
    model.save_weights("splitmodel.h5")
