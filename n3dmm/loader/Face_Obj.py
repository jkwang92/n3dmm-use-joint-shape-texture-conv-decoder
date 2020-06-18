import numpy as np
import math
import os

class Face_Obj(object):

    def __init__(self,data_file_path):

        self.data_file_path = data_file_path
        self.data_list = []
        self.readData()
        self.data_size = self.data_list.size

    def readData(self):
        # get vertices from *.obj files
        with open(self.data_file_path) as file:
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    self.data_list.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "f":
                    break
        self.data_list = np.array(self.data_list)+np.ones((53215,3))