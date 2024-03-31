#importing dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

#loading the data
file = open("Shakespeare.txt").read()

#preprocessing (standardization by tokenization)
def tokenize_words(input):
  input = input.lower()
  Tokenizer = RegexpTokenizer(r'\w+')
  tokens = Tokenizer.tokenize(input)
  filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
  return "".join(filtered)

processed_inputs = tokenize_words(file)

#Characters to Numbers
chars = sorted(list(set(processed_inputs)))
chars_to_nums = dict((c, i) for i, c in enumerate(chars))

#print(chars_to_nums)

#to check if the conversion worked
input_length = len(processed_inputs)
vocab_length = len(chars)
print('Total number of characters: ', input_length)
print('Total vocabulary: ', vocab_length)

#sequence length
seq_length = 100
x_data = []
y_data = []

#looping through the sequence
for i in range(0, input_length - seq_length, 1):
  in_seq = processed_inputs[i:i + seq_length]
  out_seq = processed_inputs[i + seq_length]
  x_data.append([chars_to_nums[char] for char in in_seq])
  y_data.append(chars_to_nums[out_seq])

n_patterns = len(x_data)
print('Total patterns: ', n_patterns)

#converting input sequence to np array
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_length  )

#one hot encoding
y = to_categorical(y_data)

#Creating the Model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(y.shape[1], activation='softmax'))

#Compiling the Model
model.compile(optimizer='adam', loss='categorical_crossentropy')

#Saving the Weights
filepath = 'model_weights_saved.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

#Fitting and Training the Model
model.fit(X, y, batch_size=256, epochs=8, callbacks=desired_callbacks)

#Recompiling the model with the saved weights
filename = 'model_weights_saved.hdf5'
model.load_weights(filename)
model.compile(optimizer='adam', loss='categorical_crossentropy')

#output of the model back into characters
nums_to_chars = dict((i,c) for i,c in enumerate(chars))

#random seed to help generate
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print('Random seed:')
print("\"", ''.join([nums_to_chars[value] for value in pattern]), "\"")

#Generating the text
for i in range(100):
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  x = x/float(vocab_length)
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  result = nums_to_chars[index]
  seq_in = [nums_to_chars[value] for value in pattern]
  sys.stdout.write(result)
  pattern.append(index)
  pattern = pattern[l: len(pattern)]
