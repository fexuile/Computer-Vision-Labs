import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad, x_grad)
    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   
    grad_dir = grad_dir % np.pi
    output = np.zeros_like(grad_mag)

    mask_0 = (grad_dir < np.pi / 8) | (grad_dir >= 7 * np.pi / 8)
    mask_45 = (grad_dir >= np.pi / 8) & (grad_dir < 3 * np.pi / 8)
    mask_90 = (grad_dir >= 3 * np.pi / 8) & (grad_dir < 5 * np.pi / 8)
    mask_135 = (grad_dir >= 5 * np.pi / 8) & (grad_dir < 7 * np.pi / 8)

    compare_left = np.roll(grad_mag, 1, axis=1)
    compare_right = np.roll(grad_mag, -1 , axis=1)
    output[mask_0 & (grad_mag >= compare_left) & (grad_mag >= compare_right)] = grad_mag[mask_0 & (grad_mag >= compare_left) & (grad_mag >= compare_right)]

    compare_top_right = np.roll(np.roll(grad_mag, 1, axis=1), 1, axis=0)
    compare_bottom_left = np.roll(np.roll(grad_mag, -1, axis=1), -1, axis=0)
    output[mask_45 & (grad_mag >= compare_top_right) & (grad_mag >= compare_bottom_left)] = grad_mag[mask_45 & (grad_mag >= compare_top_right) & (grad_mag >= compare_bottom_left)]

    compare_top = np.roll(grad_mag, 1, axis=0)
    compare_bottom = np.roll(grad_mag, -1, axis=0)
    output[mask_90 & (grad_mag >= compare_top) & (grad_mag >= compare_bottom)] = grad_mag[mask_90 & (grad_mag >= compare_top) & (grad_mag >= compare_bottom)]

    compare_top_left = np.roll(np.roll(grad_mag, 1, axis=1), -1, axis=0)
    compare_bottom_right = np.roll(np.roll(grad_mag, -1, axis=1), 1, axis=0)
    output[mask_135 & (grad_mag >= compare_top_left) & (grad_mag >= compare_bottom_right)] = grad_mag[mask_135 & (grad_mag >= compare_top_left) & (grad_mag >= compare_bottom_right)]

    return output


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """
    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.4375
    high_ratio = 1.065
    sum_img, total = 0, 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:
                sum_img += img[i, j]
                total += 1
    avg_img = sum_img / total
    maxVal = avg_img * high_ratio
    minVal = avg_img * low_ratio
    output = np.zeros_like(img)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= maxVal:
                output[i, j] = 1
                q = [(i, j)]
                while q:
                    x, y = q.pop()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and img[nx, ny] >= minVal and not output[nx, ny]:
                            output[nx, ny] = 1
                            q.append((nx, ny))
    return output

if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)
    write_img("result/HM1_Canny_magnitude.png", magnitude_grad*255)
    # write_img("result/HM1_Canny_direction.png", direction_grad*255)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)
    write_img("result/HM1_Canny_NMS.png", NMS_output*255)
    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
