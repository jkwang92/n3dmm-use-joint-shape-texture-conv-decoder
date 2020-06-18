import numpy as np
import os
import math


def write_obj(obj_path, content):
    with open('./f.txt') as f_file:
        f = []
        while 1:
            line = f_file.readlines()
            if not line:
                break
            f.append(line)

    with open(obj_path, 'w') as obj:
        for i in range(content.size[0]):
            point = 'v'+' '+str(content[i][0])+' '+str(content[i][1])+' '+str(content[i][2])
            obj.writelines(point)
            obj.write('\n')
        for var in f:
            obj.writelines(var)
            obj.write('\n')

