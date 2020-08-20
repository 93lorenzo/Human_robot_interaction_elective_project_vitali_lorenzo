

import argparse
import os
import shutil
import time
from builtins import input

import utils_resnet_basic
from image_dataset import ImageDataset
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

N_CLASS=3

class ResNet101_5(nn.Module):
    def __init__(self):
        super(ResNet101_5, self).__init__()

        #num_classes = 5
        num_classes = N_CLASS
        #num_classes = 1000

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

        return x, x_p


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch):

    # switch to train mode
    model.train()

    str_ = str(('{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())))
    fh_target = open(str_ + "current_target.txt", "a+")
    fh_output = open(str_ + "current_output.txt", "a+")


    for i, (input, target, file_path) in enumerate(train_loader):

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, x_p_features = model(input_var)
        #print(x_p)
        for iiii in range ( len(target) ):
            #print(str(target[iiii]) + " " + str(file_path[iiii]) + "\n")
            fh_target.write(str(target[iiii]) + " " + str(file_path[iiii]) + "\n")


        #print("len output "+str( len(output)) )
        for list_output in output :
            #print("len list output "+str(len(list_output)) )
            #for single_value in list_output :
            for single_value in list_output:
                # this is the value we have to write as the output of the net for the image
                #print( str( single_value.data.cpu().numpy()[0] ) + "," )
                fh_output.write( str(single_value.data.cpu().numpy()[0]) + "," )
            fh_output.write( "\n")
        # FROM HERE I CAN TAKE VALUES AND LABELS


        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, N_CLASS))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f} \t'
                  'Accuracy {top1:.3f} \t'.format(
                epoch, i, len(train_loader),
                loss=loss.data[0], top1=prec1[0]))

    fh_target.close()
    fh_output.close()


def validate(val_loader, model, criterion,info=""):

    # switch to evaluate mode
    model.eval()

    avg_prec1 = 0

    directory="current_batch_weights/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fh_target = open(directory+info+"current_target.txt", "w")
    fh_output = open(directory+info+"current_output.txt", "w")

    for i, (input, target, file_path) in enumerate(val_loader):
        #input = input.cuda()
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output,x_p = model(input_var)
        loss = criterion(output, target_var)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, N_CLASS))

        avg_prec1 += prec1[0]
        if i % 100 == 0:
            print('Test: [{0}/{1}]   \t'
                  'Loss {loss:.4f} \t'
                  'Accuracy {top1:.3f}\t'.format(
                   i, len(val_loader),  loss=loss.data[0],
                   top1=prec1[0]))

    avg_prec1 /= len(val_loader)
    print("Avg test accuracy: ", avg_prec1)

    fh_target.close()
    fh_output.close()
    #utils_resnet_basic.writing_on_file(avg_prec1)
    return avg_prec1,loss.data[0]
    #return avg_prec1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', filename_best="model_best.pth.tar"):
    torch.save(state, filename)
    if is_best:
        #shutil.copyfile(filename, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


def main():
    train_model =True
    dataset_path = '/home/lorenzo/HRI_DATASET/data/'

    #checkpoint_filename='model_best.pth.tar'

    model = ResNet101_5()
    lr = 0.001

    batch_size = 32
    workers = 4
    epochs = 3
    start_epoch = 0
    percentage = 0.70

    directory_checkpoint="checkpoint_hri/"
    if not os.path.exists(directory_checkpoint):
        os.makedirs(directory_checkpoint)

    checkpoint_filename=directory_checkpoint+'___Amodel_best_batch'+str(batch_size)+'_lr'+str(lr)+'_'+str(percentage)+'.pth.tar'
    filename_checkpoint_save_func =directory_checkpoint+ '___Acheckpoint_batch'+str(batch_size)+'_lr'+str(lr)+'_'+str(percentage)+'.pth.tar'

    ###
    #percentage=1.0

    # Do not fine tune the core_cnn (resnet), use it only as a features generator
    for p in model.core_cnn.parameters():
        p.requires_grad = False # do not train this parameter

    # Fine tune only the last cnn layer 4 and the fully connected layer
    for p in model.core_cnn.layer4.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True # train this parameter

    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()


    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr =lr)

    cudnn.benchmark = True

    traindir = os.path.join(dataset_path, 'train')
    valdir = os.path.join(dataset_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transformation = transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    # BEGIN create train e test files
    utils_resnet_basic.create_image_files(percentage=percentage,dataset_path=dataset_path)
    # END create train e test files

    train_file_name= "train_"+str(percentage)+".txt"
    train_dataset = ImageDataset( transform=train_transformation,split_filename=train_file_name)

    print(train_transformation)
    print(train_file_name)
    print(train_dataset.labels)

    train_sampler=None
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    #    num_workers=workers, pin_memory=True, sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, sampler=train_sampler)


    test_file_name= "test_"+str(percentage)+".txt"
    test_dataset = ImageDataset( transform=train_transformation,split_filename=test_file_name)
    #test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    #    num_workers=workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers,  sampler=train_sampler)

    print('Train dataset size:', len(train_loader.dataset))
    print('Validation dataset size:', len(test_loader.dataset))

    best_prec1 = 0
    if os.path.isfile(checkpoint_filename):
        print("=> loading checkpoint '{}'".format(checkpoint_filename))
        #checkpoint = torch.load(checkpoint_filename)
        checkpoint = torch.load(checkpoint_filename,  map_location=lambda storage, loc: storage)

        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})  accuracy: {}"
              .format(checkpoint_filename, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_filename))

    #model.cuda()
    
    if not train_model:
        validate(test_loader, model, criterion)
        return

    print(train_loader.dataset.data)

    for epoch in range(start_epoch, epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        info="epoch="+str(epoch)
        prec1,curr_loss = validate(test_loader, model, criterion,info)

        info="Batch_Size"+str(batch_size)+"_LR"+str(lr)+"_percentage"+str(percentage)
        utils_resnet_basic.writing_on_file_prec_loss(prec1,curr_loss,info)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best,filename_checkpoint_save_func ,checkpoint_filename)


if __name__ == "__main__":
    main()
