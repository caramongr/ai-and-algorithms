from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

number_of_words = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

# print(X_train[0])

word_to_index = imdb.get_word_index()

# print(word_to_index)

# print(word_to_index['corporatism'])

index_to_word = {index: word for (word, index) in word_to_index.items()}

[print(index_to_word[i]) for i in range(1 , 52)]

words_per_view = 200

# print(X_train.shape)
# print(X_train[0])
# print(X_train[0][1])

X_train = pad_sequences(X_train, maxlen=words_per_view)

X_test = pad_sequences(X_test, maxlen=words_per_view)

print(X_train.shape)

print(X_test.shape)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=11)

print(X_train.shape)

print(X_val.shape)

from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten
from tensorflow.keras.models import Sequential

rnn = Sequential()

rnn.add(Embedding(input_dim=number_of_words, output_dim=128, input_length=words_per_view))

rnn.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))

rnn.add(Dense(units=1, activation='sigmoid'))

rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

rnn.summary()

rnn.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val))

loss, accuracy = rnn.evaluate(X_test, y_test)

ptint(f'Accuracy: {accuracy:3%}')   

print(f'Loss: {loss:3%}')


