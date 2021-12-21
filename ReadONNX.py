import onnx
from onnx import numpy_helper
import numpy as np
model = onnx.load("mobilenet_v2.onnx")
initializers = model.graph.initializer
param_dict = {}

for initializer in initializers:
    W = numpy_helper.to_array(initializer)
    # print(initializer)
    param_dict[initializer.name] = W

weight_data = []
bias_data = []
param_list = list(param_dict.items())
for k, v in param_list[2:-1]:
    # print(k, len(v.shape))
    if len(v.shape) == 4:
        weight_data.append([k, v])
    elif len(v.shape) == 1:
        bias_data.append([k, v])
    else:
        print(v.shape)
# print(param_list[0][0], param_list[0][1].shape)
# print(param_list[1][0], param_list[1][1].shape)
weight_data.append([param_list[0][0], param_list[0][1][:, :, np.newaxis, np.newaxis]])
bias_data.append([param_list[1][0], param_list[1][1]])

# with open("./weight_data.txt", "w") as w1, open("./bias_data.txt", "w") as w2:
#     for idx, (w_item, b_item) in enumerate(zip(weight_data, bias_data)):
#         w = w_item[1]
#         b = b_item[1]
#         w_shape = w.shape
#         b_shape = b.shape
#         print(idx, w_item[0], w_shape, b_item[0], b_shape)
#         for i in range(w_shape[0]):
#             for j in range(w_shape[1]):
#                 for k in range(w_shape[2]):
#                     for t in range(w_shape[3]):
#                         w1.write(str(w[i][j][k][t]) + ' ')
#             w2.write(str(b[i]) + ' ')
#         w1.write('\n')
#         w2.write('\n')
