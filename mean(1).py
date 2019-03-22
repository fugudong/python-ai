
import pandas as pd
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from itertools import chain
import tensorflow as tf

class Equal_depth_box:

    def equal_box(self, list, bin_num):
        '''
        param:
        list:you need bin box list
        bin_num: you want bin num
        '''
        list.sort()  # need sort can replace by others
        list_2 = list.copy()
        all_num = len(list_2)
        bin_sep = all_num / bin_num
        if(-int(bin_sep)+bin_sep):
            bin_num += 1
        bin_sep = int(bin_sep)
        bin_list = []
        print("all_num bin_num bin_sep",all_num, bin_num, bin_sep)
        bin_real_array = np.zeros((bin_num, bin_sep), dtype=np.int)
        for i in range(0, bin_num):
            # print(i)
            for j in range(0, bin_sep):
                if((i*bin_sep+j)<all_num):
                    bin_real_array[i][j] = list[i*bin_sep+j]
        return (bin_real_array)

    def mean_box(self, bin_data):
        '''
        param:
        list_1:you need bin list
        replace_list: from equal box, replace the original list
        '''
        bin_num = bin_data.shape[0]
        bin_step = bin_data.shape[1]
        bin_sum = np.zeros(bin_num, dtype=np.int)
        bin_last_len = 0
        for i in range(0, bin_num):
            for j in range(0, bin_step):
                bin_sum[i] += bin_data[i][j]
                if (i==(bin_num-1)) and (bin_data[i][j]):
                    bin_last_len += 1
            if i==(bin_num-1):
                bin_sum[i] = int(bin_sum[i] / bin_last_len)
            else:
                bin_sum[i] = int(bin_sum[i]/bin_step)
        for i in range(0, bin_num-1):
            for j in range(0, bin_step):
                bin_data[i][j] = bin_sum[i]
        for j in range(0, bin_last_len):
            bin_data[bin_num-1][j] = bin_sum[bin_num-1]
       # print("mean_data", bin_data)
        return bin_data

    def border_box(self, data):
        bin_num = data.shape[0]
        bin_step = data.shape[1]
        bin_last_len = 0
        for j in range(0, bin_step):
            if (data[bin_num - 1][j]):
                bin_last_len += 1
        for i in range(0, bin_num-1):
            for j in range(0, bin_step):
                if ((data[i][j]-data[i][0])>(data[i][bin_step-1] - data[i][j])):
                    data[i][j] = data[i][bin_step-1]
                else:
                    data[i][j] = data[i][0]
        for j in range(0, bin_last_len):
            if ((data[bin_num-1][j] - data[bin_num-1][0]) >
                    (data[bin_num-1][bin_last_len - 1] - data[bin_num-1][j])):
                data[bin_num-1][j] = data[bin_num-1][bin_last_len - 1]
            else:
                data[bin_num-1][j] = data[bin_num-1][0]
       # print("box_data", data)
        return data

if __name__ == '__main__':

    bin_class = Equal_depth_box()

    list_data = [5, 7,20, 9 ,12 ,14 ,16, 50,26,32 ,36, 40] #= random.sample(range(50), 16)
    original_data = copy.deepcopy(list_data)
    print("original list", original_data);
    sort_index = np.argsort(list_data)
    print("sort_index", sort_index)
    list_data.sort()
    print('sorted list: {}'.format(list_data[0:50]))

    bin_data = bin_class.equal_box(list_data, 3)
    temp = copy.deepcopy(bin_data)
    mean_data = list(chain(*bin_class.mean_box(temp)))
    print("mean_data", mean_data)

    temp = copy.deepcopy(bin_data)
    border_data = list(chain(*bin_class.border_box(temp)))
    print("border_data", border_data)

    original_mean_data = np.zeros((len(list_data)), dtype=np.int)
    original_border_data = np.zeros((len(list_data)), dtype=np.int)
    for i in range(0, len(list_data)):
        original_mean_data[sort_index[i]] = mean_data[i]
        original_border_data[sort_index[i]] = border_data[i]

    x = np.linspace(0, 1, len(list_data))
    plt.subplot(211)
    plt.plot(x, original_data[0:len(list_data)],marker='o', label='orginal')
    plt.plot(x, original_mean_data[0:len(list_data)], marker='*', label='mean')
    plt.plot(x, original_border_data[0:len(list_data)],marker='*', label='border')
    plt.title("original");
    plt.subplot(212)
    plt.plot(x, list_data[0:len(list_data)], marker='o', label='orginal')
    plt.plot(x, mean_data[0:len(list_data)], marker='*',label='mean')
    plt.plot(x, border_data[0:len(list_data)], marker='*',label='border')
    plt.title("sorted");
    plt.legend()
    plt.show()