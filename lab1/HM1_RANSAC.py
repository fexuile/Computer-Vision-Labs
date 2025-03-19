import numpy as np
from utils import draw_save_plane_with_points, normalize

def fit_plane_vectorized(points):
    p1, p2, p3 = points[:, 0], points[:, 1], points[:, 2]
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)  # 计算法向量 (sample_time, 3)
    A, B, C = normal[:, 0], normal[:, 1], normal[:, 2]
    D = -np.sum(normal * p1, axis=1)  # 计算 D (sample_time,)
    return np.column_stack((A, B, C, D))  # 返回平面方程 (sample_time, 4)

def perpendicular_distance_vectorized(planes, points):
    A, B, C, D = planes[:, 0], planes[:, 1], planes[:, 2], planes[:, 3]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    numerator = np.abs(A[:, None] * x + B[:, None] * y + C[:, None] * z + D[:, None])  # (sample_time, num_points)
    denominator = np.sqrt(A**2 + B**2 + C**2)[:, None]  # (sample_time, 1)
    return numerator / denominator  # (sample_time, num_points)

if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     

    sample_time = 9 # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    distance_threshold = 0.05

    # sample points group

    num_points = noise_points.shape[0]
    sample_indices = np.random.choice(num_points, (sample_time, 3), replace=False)
    sample_points = noise_points[sample_indices]

    # estimate the plane with sampled points group

    planes = fit_plane_vectorized(sample_points)


    #evaluate inliers (with point-to-plance distance < distance_threshold)
    distances = perpendicular_distance_vectorized(planes, noise_points)
    inliers = distances <= distance_threshold
    inlier_counts = np.sum(inliers, axis=1)
    sum_squared_distances = np.sum((distances * inliers) ** 2, axis=1)


    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    pf_idx = np.argmax(inlier_counts * 1e6 - sum_squared_distances)  # 优先 inlier_count，其次 sum_squared_distances
    pf = planes[pf_idx]

    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
