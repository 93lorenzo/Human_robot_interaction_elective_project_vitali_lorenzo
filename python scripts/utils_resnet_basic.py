import os
import matplotlib.pyplot as plt
import glob
import random
directory = "test_values/"


def writing_on_file_prec_loss(prec,loss,info):
    # crete the folder with the results
    directory="test_values/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    #FILENAME=directory+"test_plot.txt"

    # PREC
    FILENAME = directory + info + "prec_plot.txt"
    writing_mode = "a+"
    file = open(FILENAME , writing_mode)
    file.write(str(prec)+"\n")
    file.close()

    # LOSS
    FILENAME = directory + info + "loss_plot.txt"
    writing_mode = "a+"
    file = open(FILENAME , writing_mode)
    file.write(str(loss)+"\n")
    file.close()


def making_plot():
    directory="test_values/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    #FILENAME=directory+"test_plot.txt"
    #FILENAME = directory + "Batch_Size32_LR0.001loss_plot.txt"
    batch_size=256
    lr=0.001
    percentage=0.8
    #FILENAME = directory + "Batch_Size"+str(batch_size)+"_LR"+str(lr)+"_percentage"+str(percentage)+"loss_plot.txt"
    FILENAME = directory + "Batch_Size" + str(batch_size) + "_LR" + str(lr) + "_percentage" + str(
        percentage) + "prec_plot.txt"

    if not os.path.isfile(FILENAME):
        print("NO PLOT, FILE DOES NOT EXIST")
    else :
        # read input_values
        with open(FILENAME) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

        plt.plot(content)
        plt.ylabel('Input values')
        plt.xlabel('Epochs')
        plt.show()

def create_image_files(dataset_path="",percentage=""):
    # init
    if(dataset_path==""):
        #dataset_path = "/home/lorenzo/PycharmProjects/Street_Dataset/"
        dataset_path = "/home/lorenzo/Street_Dataset/"
    if(percentage==""):
        percentage= 0.7

    class_data=[]
    class_names = ([x[0] for x in os.walk(dataset_path)])

    for label_dir in class_names :
        class_data.append( (glob.glob(label_dir+"/*.png")) )
        print(class_data)
    class_data.remove([])
    # mix the vectors
    random.shuffle(class_data)

    for c in class_data:
        print (c)
    print("SIZE CLASSE DATA"+str(len(class_data)))
    class_data=list(filter(len,class_data))
    #class_data = list(filter([], class_data))
    print("SIZE CLASSE DATA "+str(len(class_data)))

    # take a small percentage
    class_data_train = list()
    class_data_test  = list()
    for single_class in class_data :
        index = round( len(single_class)*percentage )
        class_data_train.append([str(x) for x in  single_class[0 : index ] ])
        class_data_test.append ([str(x) for x in  single_class[index : -1] ])
    ###
    # train test list made
    ###

    f_train = open("train_" + str(percentage) + ".txt", "w")
    f_test =  open("test_"  + str(percentage) + ".txt", "w")

    print("here I write the train label path file")

    for i in range(len(class_data)):
        # train
        for fname_train in class_data_train[i]:
            f_train.write( str(fname_train ) + " " + str(i) + "\n")
        # test
        for fname_test in class_data_test[i]:
            f_test.write( str(fname_test  ) + " " + str(i) + "\n")

    f_train.close()
    f_test.close()

#create_image_files()
#making_plot()