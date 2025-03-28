import numpy as np
from utils import draw_save_plane_with_points, normalize

def fit_plane_vectorized(points):
    p1, p2, p3 = points[:, 0], points[:, 1], points[:, 2]
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    A, B, C = normal[:, 0], normal[:, 1], normal[:, 2]
    D = -np.sum(normal * p1, axis=1)
    return np.column_stack((A, B, C, D))

def perpendicular_distance_vectorized(planes, points):
    A, B, C, D = planes[:, 0], planes[:, 1], planes[:, 2], planes[:, 3]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    numerator = np.abs(A[:, None] * x + B[:, None] * y + C[:, None] * z + D[:, None])  # (sample_time, num_points)
    denominator = np.sqrt(A**2 + B**2 + C**2)[:, None]
    return numerator / denominator

def fit_plane_least_squares(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    cov_matrix = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]

    # Compute D
    A, B, C = normal
    D = - (A * centroid[0] + B * centroid[1] + C * centroid[2])
    return np.array([A, B, C, D])

if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     

    sample_time = int(np.ceil(np.log(1 - 0.999) / np.log(1 - (100 / 130) ** 3)))
     # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
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
    idx = np.argmax(inlier_counts * 1e6 - sum_squared_distances)
    inlier_points = noise_points[inliers[idx]]
    pf = fit_plane_least_squares(inlier_points)

    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
