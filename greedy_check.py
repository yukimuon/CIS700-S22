import numpy as np
from tensorflow import keras
from matplotlib import pyplot
import threading
import pickle
import argparse
import logging
import os
import random
logging.basicConfig(handlers=[logging.StreamHandler()],
                    # logging.basicConfig(handlers=[logging.FileHandler("admnist.log"), logging.StreamHandler()],
                    format='%(asctime)s - %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def save_img(img, i):
    pyplot.imshow(img, cmap=pyplot.get_cmap('gray'))
    pyplot.savefig(f'images/{i}.png')


with open("points.pickle",'rb') as pf:
    data = pickle.load(pf)

data = {k:v for k,v in data.items() if len(list(set(v)))>0}
print(len(data))

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.models.load_model('mnist.h5')
logging.info("Training data processing")
x_processed = x_train.astype("float32") / 255
logging.info("Training data processed")

random_order_data = {}

for l in data.keys():
    k = 0
    onex = x_processed[l]
    random_point = random.choice(data[l])
    onex[random_point[0]][random_point[1]] = 1 - onex[random_point[0]][random_point[1]]

    random_point0 = random_point
    while random_point0 == random_point:
        random_point0 = random.choice(data[l])
    onex[random_point0[0]][random_point0[1]] = 1 - onex[random_point0[0]][random_point0[1]]

    bestpos = [random_point, random_point0]

    while model.predict(np.expand_dims(onex, axis=0))[0].argmax() == y_train[l] and k <= 20:
        lowest = model.predict(np.expand_dims(onex, axis=0))[0][y_train[l]]
        bestpos.append((0, 0))
        for i in range(28):
            for j in range(28):
                another = onex.copy()
                another[i][j] = 1-another[i][j]
                preds = model.predict(np.expand_dims(another, axis=0))
                if y_train[l] != preds[0].argmax():
                    lowest = preds[0][y_train[l]]
                    bestpos[-1] = (i, j)
                elif preds[0][y_train[l]] < lowest:
                    lowest = preds[0][y_train[l]]
                    bestpos[-1] = (i, j)
        onex[bestpos[-1][0]][bestpos[-1][1]] = 1 - onex[bestpos[-1][0]][bestpos[-1][1]]
        logging.info(f"Changed {k} points on image {l}, now recognized as " +
                        str(model.predict(np.expand_dims(onex, axis=0))[0].argmax()) +
                        " with confidence: " +
                        str(max(model.predict(np.expand_dims(onex, axis=0))[0])))
        k += 1
    random_order_data[l] = bestpos
    file_to_store = open(f"random_order.pickle", "wb")
    pickle.dump(random_order_data, file_to_store)
    file_to_store.close()
    print(random_order_data)
    