# https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb
# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.1-text-generation-with-lstm.ipynb

import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, CuDNNLSTM, Dropout


# Generate and return Shakespearean text
def generate(numChars=1000):
    global seedText
    
    output = ""
    for i in range(numChars):
        sampled = np.zeros((1, maxInputLength, vocabSize))
        for j, char in enumerate(seedText):
            sampled[0, j, charIndices[char]] = 1.0

        predictions = model.predict(sampled, verbose=0)[0]
        nextIndex = sample(predictions)
        nextChar = uniqueChars[nextIndex]

        seedText += nextChar
        seedText = seedText[1:]

        output += nextChar
    return output


def newModel():
    model = Sequential()
    model.add(CuDNNLSTM(128, input_shape=(maxInputLength, vocabSize)))
    model.add(Dense(vocabSize, activation="softmax"))
    
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    return model


def sample (predictions, temperature=0.5):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature
    expPredictions = np.exp(predictions)
    predictions = expPredictions / np.sum (expPredictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

# Perhaps converting from Unicode to ASCII can improve performance?


filePath = "shakespeare_training.txt"
text = open(filePath).read()
textFileLength = 1115394 # pre-calculated as length of the file is static

maxInputLength = 60  # max sequence length
    
# Every unique character in order in a list
# uniqueChars = sorted(set(text))
# vocabSize = len(uniqueChars)

uniqueChars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
vocabSize = 65

charIndices = {i:c for c, i in enumerate(uniqueChars)}
    
# print("Unique chars: %d" % vocabSize)
# print(uniqueChars)
    
# model = newModel()
model = tf.keras.models.load_model("shakespeareModel.h5")
# model.fit(trainData, trainTargets, epochs=12, batch_size=128)
# model.save("shakespeareModel")

with open(filePath) as textFile:
    textFile.seek(random.randrange(0, 1115394-maxInputLength))
    seedText = textFile.read(maxInputLength)

# seedIndex = random.randrange(0, len(text) - maxInputLength)
# seedText = text[seedIndex : seedIndex + maxInputLength]
print("Generating text with seed:", seedText)
    
train = False
    
if train:
       
    print("Vectorization... (May take a moment)")
    
    # our extracted text and its targets
    sentences, nextChars = [], []
    
    step = 3  # number of characters to skip for every sequence
    
    # append sequences of characters into sentences and the following character in nextChars
    for i in range (0, len(text) - maxInputLength, step):
        sentences.append(text[i:i + maxInputLength])
        nextChars.append(text[i + maxInputLength])
        
    print("Num of sequences: %d" % len(sentences))
    
    '''
    One-hot encoding of data into binary arrays
    for example, if characters a,b,c are the possible characters,
    a = [1,0,0], b = [0,1,0], c = [0,0,1]
    so a 1 shows that it is a certain type (character)
    '''
    # create arrays that are initialized with all 0s
    trainData = np.zeros((len(sentences), maxInputLength, vocabSize), dtype=np.bool)
    trainTargets = np.zeros((len(sentences), vocabSize), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            trainData[i, t, charIndices[char]] = 1
        trainTargets[i, charIndices[nextChars[i]]] = 1

"""
# Number of characters to generate
outputLength = 1000

fileIndex = 1

while os.path.isfile("shakespeareGenerated" + str(fileIndex) + ".txt"):
    fileIndex += 1
outputFileName = "shakespeareGenerated" + str(fileIndex) + ".txt"

outputFile = open(outputFileName, 'w')

print("\nSHAKESPEARE SUCCESSFULLY GENERATED TO", outputFileName + "\n")

# We generate (outputLength) characters
for i in range(outputLength):
    sampled = np.zeros((1, maxInputLength, vocabSize))
    for j, char in enumerate(seedText):
        sampled[0, j, charIndices[char]] = 1.0

    predictions = model.predict(sampled, verbose=0)[0]
    nextIndex = sample(predictions)
    nextChar = uniqueChars[nextIndex]

    seedText += nextChar
    seedText = seedText[1:]

    print(nextChar, end="", file=outputFile)

    
    
outputFile.close()
"""
        
