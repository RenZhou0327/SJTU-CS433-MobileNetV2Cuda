import numpy as np


def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # print(level1)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # print(everyLevels)
    # exit(0)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def img2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)

    # x_list = []
    # for m in range(0, 27):
    #     for n in range(0, 4):
    #         x = d[m]
    #         y = i[m, n]
    #         z = j[m, n]
    #         x_list.append(int(X_padded[0, x, y, z]))
    # x_list = np.array(x_list).reshape((27, 4))
    # print(x_list)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]

    cols = np.concatenate(cols, axis=-1)
    # print(cols.shape)
    # print(np.array_equal(x_list, cols))
    # exit(0)
    return cols


def forward(X, W, b, p, s):
    """
        Performs a forward convolution.

        Parameters:
        - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
        - W : Conv Kernel (N, C, H, W)
        Returns:
        - out: previous layer convolved.
    """
    m, n_C_prev, n_H_prev, n_W_prev = X.shape
    k_n, k_c, k_h, k_w = W.shape

    n_C = k_n
    n_H = int((n_H_prev + 2 * p - k_h) / s) + 1
    n_W = int((n_W_prev + 2 * p - k_w) / s) + 1
    print(n_H, n_W)

    X_col = img2col(X, k_h, k_w, s, p)
    w_col = W.reshape((k_n, -1))
    b_col = b.reshape(-1, 1)
    # Perform matrix multiplication.
    # out = w_col @ X_col
    out = w_col @ X_col + b_col
    # Reshape back matrix to image.
    out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
    cache = X, X_col, w_col, b_col
    return out, cache


if __name__ == '__main__':
    pass

