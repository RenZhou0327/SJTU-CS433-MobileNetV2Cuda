# generate definitions of variables
# generate some trivial functions codes such as malloc, fread

import re


def get_length_def(length):
    print("=" * 50)
    for idx in range(1, length + 1):
        print(f"extern const int w{idx}_len, b{idx}_len;")


def get_length_value(weight_list, bias_list):
    print("=" * 50)
    for idx, (w, b) in enumerate(zip(weight_list, bias_list)):
        print(f"const int w{idx + 1}_len = {w[0]} * {w[1]} * {w[2]} * {w[3]}, b{idx + 1}_len = {b[0]};")


def get_array_def(length):
    print("=" * 50)
    for idx in range(1, length + 1):
        print(f"float *w{idx}, *b{idx};")
        # print(f"float w{idx}[w{idx}_len], b{idx}[b{idx}_len];")


def get_weight_read(length):
    print("=" * 50)
    for idx in range(1, length + 1):
        print(f"fread(w{idx}, w{idx}_len * sizeof(float), 1, w_in);", end=" ")
        print(f"fread(b{idx}, b{idx}_len * sizeof(float), 1, b_in);")


# def get_bias_read(length):
#     print("=" * 50)
#     for idx in range(1, length + 1):
#         print(f"fread(b{idx}, b{idx}_len * sizeof(float), 1, b_in);")


def get_test_data(length):
    print("=" * 50)
    for idx in range(1, length + 1):
        print(f'printf("%f %f\\n", w{idx}[w{idx}_len - 1], b{idx}[b{idx}_len - 1]);')


def get_alloc_mem(length):
    print("=" * 50)
    for idx in range(1, length + 1):
        print(f'w{idx} = (float*) malloc(w{idx}_len * float_size);', end=" ")
        print(f'b{idx} = (float*) malloc(b{idx}_len * float_size);')


def get_move_item(length):
    print("=" * 50)
    for idx in range(1, length + 1):
        print(f'move_item(&w{idx}, w{idx}_len, &b{idx}, b{idx}_len);')


def free_item(length):
    print("=" * 50)
    for idx in range(1, length + 1):
        print(f'err = cudaFree(w{idx}); assert(err == cudaSuccess);', end=" ")
        print(f'err = cudaFree(b{idx}); assert(err == cudaSuccess);')


def init_array():
    pattern = re.compile(r'\(.*?\)', re.S)
    file_path = "standard_data/dim_record.txt"
    fp = open(file_path, "r")
    lines = fp.readlines()
    fp.close()

    weight_list = []
    bias_list = []

    for line in lines:
        line = line.strip('\n')
        item_list = re.findall(pattern, line)
        weight_shape = eval(item_list[0])
        bias_shape = eval(item_list[1])
        weight_list.append(weight_shape)
        bias_list.append(bias_shape)

    length = len(weight_list)
    # get_move_item(length)
    free_item(length)


if __name__ == '__main__':
    init_array()
