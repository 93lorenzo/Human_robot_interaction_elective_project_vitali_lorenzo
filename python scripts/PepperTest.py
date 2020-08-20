#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np
import sklearn
from sklearn.externals import joblib
import scipy.ndimage
import scipy.misc
from sklearn.decomposition import PCA
import imutils
from skimage import exposure
from skimage import feature
import glob, os
import math
from scipy.ndimage import morphology
import sys

# from keras.models import model_from_json
# from keras.models import Sequential

from time import gmtime, strftime

import torch
import torch.nn as nn
from torchvision import models, transforms

# from PIL import Image
import PIL
sys.path.append('/home/lorenzo/modim/modim/src/GUI')
from ws_client import *

FAR = 0
CLOSE = 1
EMPTY = 2
N_CLASS = 3


def begin_action_local():
    begin()
    im.setPath("/home/lorenzo/modim/modim/prova_demo_lorenzo/eurobotics/")
    # im.setPath(PATH_ACTIONS)
    im.init()

def begin_action_pepper():
    begin()
    im.setPath("/home/nao/spqrel_app/html/demo/1526110")
    #im.setPath("/home/nao/spqrel_app/html/demo/cocktail_party_demo/")
    # im.setPath(PATH_ACTIONS)
    im.init()


###############
# PEPPER VAR #
###############

pepper=True
#pepper_ip = '192.168.1.134' # ethernet
pepper_ip = '10.0.1.201' # wireless
pepper_port = 9101
PEPPER_PATH = '/home/nao/spqrel_app/html/demo/cocktail_party_demo/'
PATH_ACTIONS="/home/lorenzo/modim/modim/prova_demo_lorenzo/eurobotics/"

if pepper:
    setServerAddr(pepper_ip, pepper_port)
    PATH_ACTIONS = PEPPER_PATH
    run_interaction(begin_action_pepper)

else:
    setServerAddr('127.0.0.1', 9101)
    run_interaction(begin_action_local)


###############
# PEPPER VAR #
###############


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
        # print(x)
        return x, x_p


# 0 = far 1 = close 2 = empty


# subscribers
sub = 0
# save flag
save = 0
# saving path
PATH = "/home/lorenzo/"
minimum_distance = 10000.0
# size classifier image for human legs
size_h = [75, 101]
# bridge to convert images
bridge = CvBridge()
# initialize list of boxes
rects_far = []
rects_close = []
detection = []
# counters
cont = -1
# images acquired
GRAY = np.zeros((640, 480), dtype=np.int)
depthMap = np.zeros((640, 480), dtype=np.int)
# flag
images_ok = 0

directory = strftime("%Y-%m-%d %H:%M:%S", gmtime())
if not os.path.exists("/home/lorenzo/" + directory):
    os.makedirs("/home/lorenzo/" + directory)

print directory

print("laod model")

# ********************************************
# ****** To Load the net correct name  *******
# ********************************************
lr = 0.001
batch_size = 32
percentage = 0.70
pre_path = "/home/lorenzo/Convolutional/"
directory_checkpoint = pre_path + "checkpoint_hri2/"
if not os.path.exists(directory_checkpoint):
    os.makedirs(directory_checkpoint)
# find the correct checkpoint name as we named in the training
checkpoint_filename = directory_checkpoint + '___Amodel_best_batch' + str(batch_size) + '_lr' + str(lr) + '_' + str(
    percentage) + '.pth.tar'

# ********************************************
# ********************************************
# ********************************************
# pre_path="/home/lorenzo/Convolutional/"
# file_name_correct = os.path.dirname(os.path.realpath(__file__)) + "/" + checkpoint_filename
file_name_correct = "/" + checkpoint_filename

# use the cprrect model
model = ResNet101_5()
checkpoint = torch.load(file_name_correct, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
classifier = model
classifier.eval()

W_ = 120
H_ = 240




def callbackHuman():
    global cont
    global rects_far
    global rects_close
    # obtained image
    gray = GRAY.copy()
    #cv2.imshow("final", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print("gray image is " +str(gray.shape))
    print(gray)
    # ok the image is correct
    print("after image open")
    classifier.eval()
    model = classifier
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transformation = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

        normalize,
    ])

    #windows = [[240, 120], [480, 240]]
    #windows = [[480, 240]]
    windows = [[ 240,120]]
    #return 1
    print windows
    bool = 2

    for elem in windows:
        prediction_list = []
        # defining support vars
        a = 0
        b = a + elem[0]
        c = 0
        d = c + elem[1]

        if a < 0 or b < 0 or c < 0 or d < 0:
            c += 30
            d += 30
            # continue
        # sliding window cycle
        while (b <= len(gray)):  # rows

            while (d <= len(gray[0])):  # columns
                # create empty matrix
                img_cropped = np.zeros((elem[0], elem[1]), dtype=np.int)

                # img_cropped = gray[a:b, c:d]
                img_cropped = cv2.resize(gray[a:b, c:d], dsize=(W_, H_), interpolation=cv2.INTER_CUBIC)
                #print("img cropped is " + str(img_cropped.shape) )
                #print(img_cropped)
                # cv2.imshow("final", (img_cropped) )
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2_im = img_cropped
                pil_im = PIL.Image.fromarray(cv2_im)
                img = pil_im
                img_out = img.convert('RGB')

                input = (train_transformation(img_out)).unsqueeze(0)

                input_var = torch.autograd.Variable(input, volatile=True)
                output, x_p = model(input_var)
                # print (output[0])
                top_3_prob, top_3_lab = torch.topk(output, 3)
                pred = (top_3_lab[0][0])

                prediction_list.append(int(pred))

                # CLOSE
                if int(pred) == 1:
                    bool = int(pred)

                # if far change the value
                if int(pred) == 0:
                    bool = int(pred)

                # updating vars
                c += 50
                d += 50

            a += 50
            b += 50
            c = 0
            d = c + elem[1]

    print(prediction_list)

    cv2.imshow("final" + str(cont), gray)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    rospy.sleep(2.)
    cont += 1

    cv2.imwrite("/home/lorenzo/" + directory + "/_ " + str(cont) + ".png", gray)
    cont += 1

    try:
        # a lot of empty values will determine an empty image
        n_empty_values = prediction_list.count(EMPTY)
        if (n_empty_values > len(prediction_list) - 2):
            return EMPTY
        # more 0 far or 1 close?
        if prediction_list.count(FAR) > prediction_list.count(CLOSE):
            return FAR
        elif prediction_list.count(FAR) <= prediction_list.count(CLOSE):
            return CLOSE


    except ValueError:
        return prediction_list[0]
        # return bool


def callbackRGB(data):
    global GRAY
    global sub
    global images_ok
    # Transform the image to a working format
    bridge = CvBridge()
    # converting image from ROS format to a working standard
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    # Converting to gray-scale
    GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print "RGB image acquired"
    sub.unregister()
    images_ok = 1

# main function
def listener():
    global sub
    global rects_far
    global rects_close
    global images_ok
    global GRAY

    # initializing listener
    rospy.init_node('stillListenerMoveV2', anonymous=True)

    # subscribe to image topics
    #sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callbackRGB)
    sub = rospy.Subscriber("/naoqi_driver/camera/front/image_raw", Image, callbackRGB)

    while images_ok == 0 and not rospy.is_shutdown():
        pass

    images_ok = 0
    # subscribe to image topics
    #sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callbackRGB)
    sub = rospy.Subscriber("/naoqi_driver/camera/front/image_raw", Image, callbackRGB)

    while images_ok == 0 and not rospy.is_shutdown():
        pass

    rects_far = []
    rects_close = []
    detection = []
    class_obtained = callbackHuman()
    if class_obtained == 0:
        print "Obtained : " + str("FAR")
        run_interaction( far_action )
    if class_obtained == 1:
        print "Obtained : " + str("CLOSE")
        run_interaction( close_action )
    if class_obtained == 2:
        print "Obtained : " + str("EMPTY")

    images_ok = 0



def close_action():
    im.execute('hello.txt')

def far_action():
    im.execute('come_here.txt')

if __name__ == '__main__':
    while not rospy.is_shutdown():
        listener()

