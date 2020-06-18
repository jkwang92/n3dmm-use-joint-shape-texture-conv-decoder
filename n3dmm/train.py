import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import argparse
from loader.Face_Obj import Face_Obj
from net.net_structure import img_encoder, pts_encoder, pts_decoder
from utils.face_recognizer import recog
from utils.write_obj import write_obj


def main(args):

    '''INITIALIZE'''
    is_training = args.is_training
    begin_index = args.begin_index
    epoch = args.epoch
    train_index_path = args.train_index_path
    eval_index_path = args.eval_index_path

    ftp = []
    obj_ftp = []
    img_ftp = []
    if is_training:
        index_path = train_index_path
        f = open(index_path)
        ftp = f.readlines()
        f.close()
    else:
        index_path = eval_index_path
        f = open(index_path + '/evaluate_obj_list.txt')
        obj_ftp = f.readlines()
        f.close()
        f = open(index_path + '/evaluate_img_list.txt')
        img_ftp = f.readlines()
        f.close()

    '''MODEL'''
    encoder = img_encoder()
    decoder = pts_decoder()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    '''SETTING'''
    optimzer_encoder = torch.optim.Adam(encoder.parameters())
    optimzer_decoder = torch.optim.Adam(decoder.parameters())
    loss_func = nn.MSELoss()

    '''FORWARD'''
    
    if not begin_index == 0:
        encoder = torch.load("./model/img_encoder.pt")
        decoder = torch.load("./model/pts_decoder.pt")

    err = []

    if is_training:
        for i in range(epoch):
            i = i+begin_index
            m = Face_Obj('/home/jikai/pixel_3d/train/train_data/obj_file/mesh'+ftp[i].strip()+'.obj')
            img = recog('/home/jikai/pixel_3d/train/train_data/image/'+ftp[i].strip()+'.jpg')
            if len(img) == 0:
                continue
                
            img = torch.from_numpy(img)
            img = torch.reshape(img,(1,3,112,112))
            mi = torch.from_numpy(m.data_list)
            mi = torch.reshape(mi,(1,3,53215,1))
                
            img = img.to(device)
            mi = mi.to(device)
                
            c = img_encoder(img)
            mo = pts_decoder(c)
            cost = loss_func(mi,mo)
                
            optimzer_encoder.zero_grad()
            optimzer_decoder.zero_grad()
                
            cost.backward()
                
            optimzer_encoder.step()
            optimzer_decoder.step()
                
            print('EPOCH:'+str(i+1)+'\n'+'ERROR:'+str(cost))
            err.append(cost)
                
            torch.save(encoder, "./model/img_encoder.pt")
            torch.save(decoder, "./model/pts_decoder.pt")

    else:
        for i in range(epoch):
            i = i + begin_index
            m = Face_Obj(obj_ftp[i].strip())
            img = recog(img_ftp[i].strip())
            if len(img) == 0:
                continue
                
            img = torch.from_numpy(img)
            img = torch.reshape(img,(1,3,112,112))
            mi = torch.from_numpy(m.data_list)
            mi = torch.reshape(mi,(1,3,53215,1))
                
            img = img.to(device)
            mi = mi.to(device)
                
            c = img_encoder(img)
            mo = pts_decoder(c)
            cost = loss_func(mi,mo)

            print('EPOCH:' + str(i + 1) + '\n' + 'ERROR:' + str(cost))
            write_obj('./results/' + str(i) + '.obj', m.data_list)
            write_obj('./results/' + str(i) + '_gen.obj', out)
            err.append(cost)
    print(err)


if __name__ == '__main__':

    par = argparse.ArgumentParser(description='Nonlinear 3DMM Training')
    par.add_argument('--is_training', default=True, type=bool, help='Train model or generate *.obj file')
    par.add_argument('--begin_index', default=0, type=int, help='The begin index of data')
    par.add_argument('--epoch', default=1000, type=int, help='The number of data for training')
    par.add_argument('--train_index_path',
                     default='/home/jikai/pixel_3d/train/train_data/image/info.txt',
                     type=str,
                     help='The path of training index file')
    par.add_argument('--eval_index_path',
                     default='/home/jikai/pixel_3d',
                     type=str,
                     help='The path of evaluate index file')

    main(par.parse_args())