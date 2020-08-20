from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import glob
from glob import glob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import cv2

training_folder="/home/lorenzo/dataset_hri/TRAIN/"
test_folder="/home/lorenzo/dataset_hri/TEST/"
test_dir = "/home/lorenzo/dataset_hri/FINAL_TEST/"
H_ = 240
W_ = 120
DR_ = 0.6
K_ = 16
K_SIZE=3
CLASSES_=3

weights = ["checkpoint_hri2/_1conv_dataset_3_classes_grey_weights.h5","checkpoint_hri2/_2conv_dataset_3_classes_grey_weights.h5","checkpoint_hri2/_3conv_dataset_3_classes_grey_weights.h5","checkpoint_hri2/_4conv_dataset_3_classes_grey_weights.h5"]
models  = ["checkpoint_hri2/_1conv_dataset_3_classes_grey_model.json","checkpoint_hri2/_2conv_dataset_3_classes_grey_model.json","checkpoint_hri2/_3conv_dataset_3_classes_grey_model.json","checkpoint_hri2/_4conv_dataset_3_classes_grey_model.json"]
current_conv = 2




def train():
    classifier = Sequential()

    classifier.add(Conv2D(K_*2, (K_SIZE, K_SIZE), input_shape=(W_, H_,1), activation='relu'))
    classifier.add(Conv2D(K_*2, (K_SIZE, K_SIZE), input_shape=(W_, H_), activation='relu')) #comment if needed
    classifier.add(Conv2D(K_*2, (K_SIZE, K_SIZE), input_shape=(W_, H_), activation='relu')) #comment if needed
    classifier.add(Conv2D(K_*2, (K_SIZE, K_SIZE), input_shape=(W_, H_), activation='relu')) #comment if needed
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(DR_))

    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(DR_))
    classifier.add(Dense(units = CLASSES_, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(training_folder,
    target_size=(W_, H_),
    batch_size = 64,
    color_mode="grayscale")

    test_set = test_datagen.flow_from_directory(test_folder,
    target_size = (W_, H_),
    batch_size = 32,
    color_mode="grayscale")

    classifier.fit_generator(training_set,
    steps_per_epoch =30,
    epochs = 4,
    validation_data = test_set,
    validation_steps = 50)

    # save model and weights
    classifier.save(weights[current_conv])
    model_json = classifier.to_json()
    with open(models[current_conv], "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    print("saved" + "\n" + weights[current_conv] + "\n" + models[current_conv] )
    print(training_set.class_indices)



def confusion_matrix_():
    json_file =        open(models[current_conv], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    classifier = loaded_model
    classifier.load_weights(weights[current_conv])

    # for the confusion matrix
    predicted = []
    actual = []

    dataset_lists = glob(test_dir+"/*/")
    for data_list in dataset_lists :
        # for each photo (element)
        elements = glob(data_list+"*.png")
        for element in elements :

            current_label = int( element.split("FINAL_TEST/")[1][0] )

            # IF TO CHECK WHICH CLASS IMPORTANT
            #if current_label == 2 or current_label == 1 :
            #    continue
            test_image = image.load_img(element, grayscale=True, )
            test_image = np.array(test_image)
            predicted_label = classification(classifier,test_image)

            print "pred = " + str(predicted_label) + " act = " + str(current_label) + str(element)
            predicted.append(predicted_label)
            actual.append(current_label)

    CLASSES=["CLOSE","EMPTY","FAR"]
    plot_confusion_matrix(cm=confusion_matrix(actual, predicted), classes=CLASSES)


def classification (classifier, gray):
    windows = [ [240,120],[480, 240]]
    windows = [[480, 240]]
    # default is empty
    bool=1
    print windows
    for elem in windows:

        # defining support vars
        a = 0
        b = a + elem[0]
        c = 0
        d = c + elem[1]

        if a < 0 or b < 0 or c < 0 or d < 0:
            c += 50
            d += 50
        # sliding window cycle
        while (b <= len(gray)):  # rows
            while (d <= len(gray[0])):  # columns
                img_cropped = cv2.resize(gray[a:b, c:d], dsize=(W_, H_), interpolation=cv2.INTER_CUBIC)
                img_cropped = np.reshape(img_cropped, (1, W_, H_, 1))
                res = classifier.predict([img_cropped])

                if res[0][0] == 1:
                    bool = 0

               if res[0][2] == 1 :#or (res[0][0] == 0 and res[0][1] == 0 and res[0][2] == 0):
                    bool = 2
                    return bool

                c += 50
                d += 50

            a += 50
            b += 50
            c = 0
            d = c + elem[1]

    return bool

def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    #train()
    confusion_matrix_()



#{'CLOSE_PERSON': 0, 'EMPTY': 1, 'FAR_PERSON': 2}

