#/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/
import argparse
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--postfix',type=str,default='val')
parser.add_argument('--label_path',type=str,default='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/ImageSets/Main')
parser.add_argument('--savepath',type=str,default='/NAS_REMOTE/shaozl/MS-CAM-NAS')

if __name__ == '__main__':
    args = parser.parse_args()
    label_path = args.label_path
    print(label_path)
    #set postfix
    postfix = args.postfix
    print(postfix)
    #set class_name list
    class_name_list = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
                       'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    # initialize a dic with each filename as key and [20] vector as value
    mullables = {}
    with open(os.path.join(label_path,postfix+'.txt')) as f:
        filenames = f.readlines()
        for filename in filenames:
            filename = filename.strip()
            mullables[filename] = [0 for i in range(20)]

    #for 20 classes
    for index, class_name in enumerate(class_name_list):
        with open(os.path.join(label_path,class_name+'_'+postfix+'.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                split = line.split(' ')
                if split[1] != '-1':
                    mullables[split[0]][index] = 1
    print(mullables)
    #save the lables file
    np.save(os.path.join(args.savepath,postfix+'_mullables.npy'),mullables)