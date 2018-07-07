import argparse
import sys

from PIL import Image
import numpy as np 
from scipy import misc
import tensorflow as tf

import facenet
import align.detect_face

import os
from sklearn import neighbors

def main(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('--model', type=str, help='directory containing pretained model.', default='')
    parser.add_argument('--i', type=str, help='path of the image.')
    parser.add_argument('--ds', type=str, help='directory containing stored photos')
    #parser.add_argument('--i2', type=str, help='second image.', default='')
    parser.add_argument('--thr', type=float,
            help='threshold to recognition.', default=1.0)

    parser = parser.parse_args()

    if not (parser.model and parser.i and parser.ds):
        print('usage: --model <path for model> --ds <path for data set> --i <path for image>')
        exit(-1)

    files = os.listdir(parser.ds)
    m_classes = []
    for i in range(len(files)):
        if not files[i].startswith('.') and os.path.isdir(parser.ds + "/" + files[i]):
            m_classes.append(files[i])

    data = {}
    for i in range(len(m_classes)):
        child_files = os.listdir(parser.ds + "/" + m_classes[i])
        for child_file in child_files:
            if child_file.endswith(".png"):
                data[parser.ds + "/" + m_classes[i] + "/" + child_file] = m_classes[i]

    with tf.Session() as sess:

        image_file_placeholder = tf.placeholder(tf.string, shape=None, name='image_file')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # load model
        input_map = {'phase_train': phase_train_placeholder}
        facenet.load_model(parser.model, input_map=input_map)

        # read image using tf
        file_contents = tf.read_file(image_file_placeholder)
        image = tf.image.decode_image(file_contents, 3)
        image = (tf.cast(image, tf.float32) - 127.5) / 128.0

        images = np.empty(shape=(2, 0))
        for i, j in data.items():
            temp = sess.run(image, feed_dict={image_file_placeholder: i})
            images = np.c_[images, [temp[np.newaxis, ...], j]]
        input_image = sess.run(image, feed_dict={image_file_placeholder: parser.i})[np.newaxis, ...]

        # evaluate on test image
        graph = tf.get_default_graph()
        image_batch = graph.get_tensor_by_name("image_batch:0")
        embeddings = graph.get_tensor_by_name("embeddings:0")

        features = np.empty(shape=(2, 0))
        for i in range(images.shape[1]):
            temp_feature = sess.run(embeddings, feed_dict={phase_train_placeholder: False, image_batch: images[0, i]})
            features = np.c_[features, [temp_feature, images[1, i]]]
        input_feature = sess.run(embeddings, feed_dict={phase_train_placeholder: False, image_batch: input_image})

        knn = neighbors.KNeighborsClassifier()
        X = features[0, :]
        X = [x[0] for x in X]
        Y = features[1, :]
        
        # knn
        knn.fit(X, Y)
        distance, ind = knn.kneighbors(X=[input_feature[0]], n_neighbors=1)
        if distance[0][0] < parser.thr:
            predict = knn.predict([input_feature[0]])
            print("The image might be", predict)
            print("The distance is",distance[0][0])
        else:
            print("could not match faces in the data set.")
            print("The nearest distance is",distance[0][0])


if __name__ == '__main__':
    main(sys.argv[1:])
