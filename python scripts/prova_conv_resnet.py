import os
from PIL import Image
import numpy as np
from keras.preprocessing import image
import glob
from glob import glob
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import cv2
import torch.nn as nn
from torchvision import models, transforms


training_folder="/home/lorenzo/dataset_hri/TRAIN/"
test_folder="/home/lorenzo/dataset_hri/TEST/"
test_dir = "/home/lorenzo/dataset_hri/FINAL_TEST/"

H_ = 240
W_ = 120
N_CLASS=3



FAR=0
CLOSE=1
EMPTY=2

# ********************************************
# ****** To Load the net correct name  *******
# ********************************************
lr = 0.001
batch_size = 32
percentage = 0.70
directory_checkpoint = "checkpoint_hri2/"
if not os.path.exists(directory_checkpoint):
    os.makedirs(directory_checkpoint)
# find the correct checkpoint name as we named in the training
checkpoint_filename = directory_checkpoint + '___Amodel_best_batch' + str(batch_size) + '_lr' + str(lr) + '_' + str(
    percentage) + '.pth.tar'
# ********************************************
# ********************************************
# ********************************************


class ResNet101_5(nn.Module):
    def __init__(self):
        super(ResNet101_5, self).__init__()

        num_classes = N_CLASS
        expansion = 4
        self.core_cnn = models.resnet101(pretrained=True)
        self.fc = nn.Linear(512 * expansion, num_classes)

        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        x = self.core_cnn.layer4(x)

        x_p = self.core_cnn.avgpool(x)
        x_p = x_p.view(x_p.size(0), -1)
        x = self.fc(x_p)
        #print(x)
        return x, x_p
#0 = far 1 = close 2 = empty




def confusion_matrix_():

    file_name_correct = os.path.dirname(os.path.realpath(__file__)) + "/" + checkpoint_filename

    # use the cprrect model
    model = ResNet101_5()
    checkpoint = torch.load(file_name_correct, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    # empty lists for the confusion matrix
    predicted = []
    actual = []

    dataset_lists = glob(test_dir+"/*/")
    for data_list in dataset_lists :
        # for all the images (elements)
        elements = glob(data_list+"*.png")
        for element in elements :
            # the net has changed the correct order of the labels
            current_label = int( element.split("FINAL_TEST/")[1][0] )
            if current_label == 0:
                #close now is 1 not 0
                current_label=1
            elif current_label == 1:
                #empty now is 2 not 1
                current_label=2
            else :
                current_label=0

            #0 far
            #1 close
            #2 empty

            # IF TO CHECK WHICH CLASS IMPORTANT
            #if current_label == 2 or current_label == 1 :
            #    continue

            # ***************
            test_image = image.load_img(element, grayscale=False, )
            test_image = np.array(test_image)
            predicted_label = classification(model, test_image)
            # ***************

            print "pred = " + str(predicted_label) + " act = " + str(current_label) + str(element)
            # populate for the confusion matrix
            predicted.append(predicted_label)
            actual.append(current_label)

    CLASSES=["FAR","CLOSE","EMPTY"]
    plot_confusion_matrix(cm=confusion_matrix(actual, predicted), classes=CLASSES)


# function to make test to try with only one image
def one_image_prediction(classifier, gray):
    classifier.eval()
    model = classifier
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transformation = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    cv2_im = gray
    pil_im = Image.fromarray(cv2_im)
    img = pil_im
    img_out = img.convert('RGB')

    input = train_transformation(img_out)
    input = input.unsqueeze(0)

    input_var = torch.autograd.Variable(input, volatile=True)
    output, x_p = model(input_var)


def classification (classifier, gray):
    classifier.eval()
    model=classifier
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transformation = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    windows = [ [240,120],[480, 240]]
    windows = [[480, 240]]
    # bool used for the result it starts at 2 -> empty class
    bool=2
    print windows

    # for the windows array
    for elem in windows:
        prediction_list=[]
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
                # create empty matrix
                img_cropped = np.zeros((elem[0], elem[1]), dtype=np.int)

                # img_cropped = gray[a:b, c:d]
                img_cropped = cv2.resize(gray[a:b, c:d], dsize=(W_, H_), interpolation=cv2.INTER_CUBIC)

                cv2_im = img_cropped
                pil_im = Image.fromarray(cv2_im)
                img = pil_im
                img_out = img.convert('RGB')

                input = train_transformation(img_out)
                input = input.unsqueeze(0)

                input_var = torch.autograd.Variable(input, volatile=True)
                output, x_p = model(input_var)
                #print (output[0])
                top_3_prob,top_3_lab=torch.topk(output,3)
                pred = (top_3_lab[0][0])
                #print(pred)
                prediction_list.append(int(pred))
                # if close detected return close
                if int(pred)==1:
                    bool=int(pred)
                    #return bool

                #if far change the value
                if int(pred)==0:
                    bool=int(pred)

                # updating vars
                c += 50
                d += 50

            a += 50
            b += 50
            c = 0
            d = c + elem[1]

    print(prediction_list)

    try :
        # a lot of empty values will determine an empty image
        n_empty_values = prediction_list.count(EMPTY)
        if(n_empty_values > len(prediction_list)-3 ):
            return EMPTY
        # more 0 far or 1 close?
        if prediction_list.count(FAR) > prediction_list.count(CLOSE):
            return FAR
        elif prediction_list.count(FAR) <= prediction_list.count(CLOSE):
            return CLOSE


    except ValueError:
        return prediction_list[0]
            #return bool


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
    confusion_matrix_()



#{'CLOSE_PERSON': 0, 'EMPTY': 1, 'FAR_PERSON': 2}

