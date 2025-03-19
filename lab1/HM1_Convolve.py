import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    width = img.shape[0]
    height = img.shape[1]

    if type=="zeroPadding":
        padding_img = np.zeros((width+2*padding_size, height+2*padding_size))
        padding_img[padding_size:width+padding_size, padding_size:height+padding_size] = img
        return padding_img
    elif type=="replicatePadding":
        padding_img = np.zeros((width+2*padding_size, height+2*padding_size))
        padding_img[padding_size:width+padding_size, padding_size:height+padding_size] = img
        padding_img[0:padding_size, :] = padding_img[padding_size]
        padding_img[padding_size+width:, :] = padding_img[padding_size+width-1]
        padding_img[:, 0:padding_size] = padding_img[:, padding_size].reshape(-1,1)
        padding_img[:, padding_size+height:] = padding_img[:, padding_size+height-1].reshape(-1,1)
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    # zero padding
    img_padding = padding(img, 1, "zeroPadding")
    #build the Toeplitz matrix and compute convolution
    row_indices = np.arange(6)[:, None] * 8 + np.arange(6)
    col_indices = np.arange(3)[:, None] * 8 + np.arange(3)
    row_indices = row_indices.reshape(-1, 1) + col_indices.reshape(1, -1)
    row_indices = row_indices.reshape(-1)
    toeplitz_row = np.repeat(np.arange(36), 9)
    toeplitz_col = row_indices
    toeplitz_data = np.tile(kernel.reshape(-1), 36)
    toeplitz_matrix = np.zeros((36, 64))
    toeplitz_matrix[toeplitz_row, toeplitz_col] = toeplitz_data

    output = np.dot(toeplitz_matrix, img_padding.flatten())
    return output.reshape(6, 6)


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    #build the sliding-window convolution here
    img_size = img.shape[0]
    kernel_size = kernel.shape[0]
    output_size = img_size - kernel_size + 1
    row_indices = np.arange(img_size - kernel_size + 1)[:, None] + np.arange(kernel_size)
    col_indices = np.arange(img_size - kernel_size + 1)[:, None] + np.arange(kernel_size)
    
    windows = img[row_indices[:, None, :, None], col_indices[None, :, None, :]]
    slide_matrix = windows.reshape(output_size * output_size, kernel_size * kernel_size)

    output = np.dot(slide_matrix, kernel.ravel()).reshape(output_size, output_size)
    
    return output

def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)

    # input_array = np.array([])
    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # #task 3: convolution with sliding-window
    result_2 = convolve(padding(input_array, 1, "zeroPadding"), input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)