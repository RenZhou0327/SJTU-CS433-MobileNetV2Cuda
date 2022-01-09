from collections import OrderedDict
import onnx
import onnxruntime as ort
import numpy as np
import pickle


if __name__ == '__main__':

    ort_session = ort.InferenceSession("models/mobilenet_v2.onnx")
    org_outputs = [x.name for x in ort_session.get_outputs()]

    model = onnx.load("models/mobilenet_v2.onnx")
    for node in model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    # excute onnx
    ort_session = ort.InferenceSession(model.SerializeToString())
    outputs = [x.name for x in ort_session.get_outputs()]

    fp = open("standard_data/mobilenetInput.txt", "r")
    lines = fp.readlines()
    fp.close()
    img = lines[0].strip('\n').split(' ')
    img = [float(x) for x in img]
    img = np.array(img, dtype=np.float32)

    fp = open("standard_data/mobilenetOutput.txt", "r")
    lines = fp.readlines()
    fp.close()
    res = lines[0].strip('\n').split(' ')
    res = [float(x) for x in res]
    res = np.array(res, dtype=np.float32)

    # in_img = np.fromfile('<you path>/input_img.raw', dtype=np.float32).reshape(1, 3, 511, 511)
    in_img = img.reshape((1, 3, 244, 244))
    ort_outs = ort_session.run(outputs, {'input.1': in_img})
    ort_outs = OrderedDict(zip(outputs, ort_outs))

    model_dict = {}
    for k, v in ort_outs.items():
        model_dict[k] = v
        print(k, v.shape, v.max(), v.min())

    fp = open("params/params.bin", "rb")
    weight_data, bias_data = pickle.load(fp)
    fp.close()

    for i, (w, b) in enumerate(zip(weight_data, bias_data)):
        print(i, w[0], w[1].shape, b[0], b[1].shape)
        model_dict[w[0]] = w[1]
        model_dict[b[0]] = b[1]

    fp = open("params/model_data.bin", "wb")
    pickle.dump(model_dict, fp)
    fp.close()
