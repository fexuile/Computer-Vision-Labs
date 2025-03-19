import numpy as np
from utils import  read_img, draw_corner, write_img
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding, Gaussian_filter



def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: array
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.
    # write_img("result/HM1_HarrisCorner_input_img.png", input_img * 255)
    blur_img = Gaussian_filter(input_img)
    # write_img("result/HM1_HarrisCorner_blur_img.png", blur_img * 255)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)
    # write_img("result/HM1_HarrisCorner_x_grad.png", x_grad * 255)
    # write_img("result/HM1_HarrisCorner_y_grad.png", y_grad * 255)
    I_xx = x_grad * x_grad
    I_yy = y_grad * y_grad
    I_xy = x_grad * y_grad

    w = np.ones((window_size, window_size))
    I_xx = convolve(I_xx, w)
    I_yy = convolve(I_yy, w)
    I_xy = convolve(I_xy, w)
    
    det_M = I_xx * I_yy - I_xy ** 2
    trace_M = I_xx + I_yy
    R = det_M - alpha * trace_M ** 2
    corner_indices = np.where(R > threshold)
    corner_list = list(zip(corner_indices[0], corner_indices[1], R[corner_indices]))

    return corner_list # array, each row contains information about one corner, namely (index of row, index of col, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 3
    alpha = 0.025
    threshold = 1

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
