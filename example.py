from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# data file: data/pima-indians-diabetes.data.csv
'''

    Number of times pregnant
    Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    Diastolic blood pressure (mm Hg)
    Triceps skin fold thickness (mm)
    2-Hour serum insulin (mu U/ml)
    Body mass index (weight in kg/(height in m)^2)
    Diabetes pedigree function
    Age (years)
    Is diabetic?
'''
dataset = loadtxt('data/pima-indians-diabetes.data.csv', delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=50)

scores = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save_weights("model.h5")

