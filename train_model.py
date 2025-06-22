import os
import re
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

# 1. Daten / Beispielkorpus
#text = "dies ist ein einfacher text für das training des sprachmodells".lower()
#text = open("german_text.txt", "r", encoding = "utf8")
#corpus = text.split()

max_seq_len = 10 #max(len(x) for x in input_sequences)
n_epochs = 50


# === TEXT AUS DATEI LADEN ===
with open('german_text.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()


# df = pd.read_csv('input/Articles.csv')
# #df_head = pd.DataFrame()
# df_body = pd.DataFrame()
# #df_head['Headline'] = df['Headline'].astype(str)
# df_body['Body'] = df['Body'].astype(str)
# df = []

# raw_text = " ".join(head for head in df_body["Body"])

# === 1. PREPROCESSING ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zäöüß\s]', '', text)  # Sonderzeichen entfernen, nur Buchstaben und Leerzeichen
    text = re.sub(r'\s+', ' ', text)          # Mehrfache Leerzeichen zu einem reduzieren
    return text.strip()

text = clean_text(raw_text)
print("Preprocessed text:")
print(text)
corpus = text.split()

# 2. Tokenisierung
tokenizer = Tokenizer(lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# 3. Trainingssequenzen erstellen
input_sequences = []
for i in range(2, len(corpus)+1):
    seq = tokenizer.texts_to_sequences([" ".join(corpus[:i])])[0]
    input_sequences.append(seq)

input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

print(f'Max_seq_len={max_seq_len}')

X = input_sequences[:,:-1]
y = input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

from tensorflow.keras.callbacks import EarlyStopping

# Trainings- und Testdaten aufteilen
# Daten durchmischen und aufteilen (z. B. 85 % Training, 15 % Test)
indices = np.arange(len(X))
#np.random.seed(42)
#np.random.shuffle(indices)

split_at = int(len(X) * 0.85)
train_idx, test_idx = indices[:split_at], indices[split_at:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# 4. Modell definieren und trainieren
model = Sequential([
    Embedding(total_words, 100, input_length=max_seq_len-1),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.01),
              metrics=['accuracy'])

# history = model.fit(X, y, epochs=n_epochs, batch_size=32, verbose=1)

#early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=n_epochs,
    batch_size=32,
    validation_split=0.1,
    #callbacks=[early_stop],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

# Perplexity berechnen
perplexity = np.exp(test_loss)

print(f"📉 Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"🤖 Perplexity: {perplexity:.2f}")

# 5. Modell speichern + in TF.js konvertieren
model.save("lm_model.h5")
os.makedirs("lm_tfjs", exist_ok=True)
tfjs.converters.save_keras_model(model, "lm_tfjs")
print("✅ Modell gespeichert in 'lm_tfjs/'")


# Tokenizer speichern
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)  # ❗️write statt json.dump

# Tokenizer Dicts direkt speichern -> funktioniert besser
with open( 'tokenizer_word2index.json' , 'w' ) as file:    
    json.dump(tokenizer.word_index, file)

with open( 'tokenizer_index2word.json' , 'w' ) as file:    
    json.dump(tokenizer.index_word, file)

print(f"Number of words in Tokenizer {len(tokenizer.word_index)}")

# 6. Testing

print("---- Prediction ----")
testing_text = "kinder spielen und die sonne"
sequence = tokenizer.texts_to_sequences([testing_text])[0]
padded = pad_sequences([sequence], maxlen=max_seq_len-1, padding='pre')
pred = model.predict(padded)
predicted_index = np.argmax(pred)
predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_index]
print(f"Vorhergesagtes Wort: {predicted_word}")


