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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainning hock")
    parser.add_argument('start', metavar='startnum', type=str,
                        help="startnum, 10000 per process")
    arg = parser.parse_args()

    attacks = {}
    logging.info("Inited")
    if os.path.isfile(f"attack_points{arg.start}.pickle"):
        with open(f"attack_points{arg.start}.pickle", 'rb') as f:
            attacks = pickle.load(f)
    logging.info(attacks.keys())

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    model = keras.models.load_model('mnist.h5')
    logging.info("Training data processing")
    x_processed = x_train.astype("float32") / 255
    logging.info("Training data processed")

    start = int(arg.start)*10000
    end = int(arg.start)*10000+10000
    for l in range(start, end):
        if l in attacks.keys():
            continue
        save_img(x_processed[l], str(l))
        onex = x_processed[l]
        bestpos = []
        k = 1
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
                    # logging.info("Img %s: %s %s flipped, now predicted as %s with confidence %s, lowest is %s",
                    #              l, i, j, preds[0].argmax(
                    #              ), preds[0][preds[0].argmax()], lowest
                    #              )
            onex[bestpos[-1][0]][bestpos[-1][1]] = 1 - \
                onex[bestpos[-1][0]][bestpos[-1][1]]
            logging.info(f"Changed {k} points on image {l}, now recognized as " +
                         str(model.predict(np.expand_dims(onex, axis=0))[0].argmax()) +
                         " with confidence: " +
                         str(max(model.predict(np.expand_dims(onex, axis=0))[0])))
            save_img(onex, "img_"+str(l)+"_"+str(k))
            k += 1
        attacks[l] = bestpos
        file_to_store = open(f"attack_points{arg.start}.pickle", "wb")
        pickle.dump(attacks, file_to_store)
        file_to_store.close()
