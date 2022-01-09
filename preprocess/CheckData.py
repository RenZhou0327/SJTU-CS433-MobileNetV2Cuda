# check output from my mobilenet
# err = Y - output, abs(err) <= 1e-5

import numpy as np
import pickle


def check_data(model_dict, layer_name):
    file_name = layer_name + ".txt"
    fp = open("layer_output/" + file_name, "r")
    read_nums = fp.read().strip('\n').split(' ')
    layer_out = np.array([float(num) for num in read_nums[:-1]])
    fp.close()
    standard_out = model_dict[layer_name]
    shape = standard_out.shape
    standard_out = standard_out.flatten()
    print(shape, standard_out.shape, layer_out.shape)
    err = standard_out - layer_out
    print(err.max(), err.min())


if __name__ == '__main__':

    # model_dict, contains W, X, B, Y
    fp = open("params/model_data.bin", "rb")
    model_dict = pickle.load(fp)
    fp.close()

    layer_name = "522"
    check_data(model_dict, layer_name)
