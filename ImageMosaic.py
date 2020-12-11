# * coding: utf8 *
import cv2
import numpy as np
import os
from numba import jit

# Adjustable ['translation', 'rotate-scale']
FOLDER_NAME = 'rotate-scale'
# Adjustable
IMAGE_SET_NAME = '1'
# Adjustable ['SIFT', 'SURF','ORB']
FEATURE_TYPE = 'SIFT'
# Adjustable ['forward', 'center','backward']
STITCH_TYPE = 'center'
# Adjustable
# 距离阈值，越接近于0，匹配程度越高，匹配数相应降低（★过小可能导致matrix为空）
DIST_THRESHOLD = 0.6

np.set_printoptions(threshold=np.inf)


# 加载待拼接的图像序列
def load_split_images():
    dir_path = './dataset/{0}/{1}'.format(FOLDER_NAME, IMAGE_SET_NAME)
    pics = os.listdir(dir_path)
    _images = []
    for pic in pics:
        _img = cv2.imread(os.path.join(dir_path, pic))
        if FOLDER_NAME == 'translation':
            _img = cv2.rotate(_img, cv2.ROTATE_90_CLOCKWISE)
        _images.append(_img)
    return _images


# 提取图像特征点
def extract_feature(img):
    # 创建描述子提取引擎
    if FEATURE_TYPE == 'SIFT':
        des_engine = cv2.xfeatures2d.SIFT_create()
    elif FEATURE_TYPE == 'SURF':
        des_engine = cv2.xfeatures2d.SURF_create()
    elif FEATURE_TYPE == 'ORB':
        des_engine = cv2.ORB_create()
    else:
        des_engine = None
    # 获取图像描述子（特征向量）
    key_points, descriptors = des_engine.detectAndCompute(img, None)
    return key_points, descriptors


# 获取最优匹配
def get_nice_matches(_des1, _des2):
    bf = cv2.BFMatcher()
    # 执行KNN最邻近匹配
    _matches = bf.knnMatch(_des1, _des2, k=2)
    # 继续处理matches，优化匹配
    nice_matches = []
    for kp1, kp2 in _matches:
        if kp1.distance < DIST_THRESHOLD * kp2.distance:
            nice_matches.append([kp1])
    return nice_matches


# 从matches中提取最优特征匹配的关键点坐标
def matches2points(kps1, kps2, matches):
    kps1_matched, kps2_matched = None, None
    for match in matches:
        point1 = np.array(kps1[match[0].queryIdx].pt, dtype=np.float32)
        point2 = np.array(kps2[match[0].trainIdx].pt, dtype=np.float32)
        kps1_matched = point1 if kps1_matched is None else np.vstack((kps1_matched, point1))
        kps2_matched = point2 if kps2_matched is None else np.vstack((kps2_matched, point2))
    return kps1_matched, kps2_matched


# 优化两张图片的衔接处
@jit
def optimize_seam(img_below, img_above, offset_x, offset_y, blend_width, stitch_type):
    m1, n1, _ = np.shape(img_below)
    m2, n2, _ = np.shape(img_above)
    # 融合宽度(★可调整·注意大小不能过大)
    blend_width = 100
    blend_height = 50
    # 目标图像大小需要设置的稍微大些
    dst_img = np.zeros((m1 + 1000, n1 + 1000, _), dtype=np.uint8)
    if stitch_type == 'left-right':
        dst_img[0:m1, 0:n1] = img_below
        # 可直接将img_above覆盖在dst_img的对应区域
        # 即：dst_img[offset_y:offset_y + m2, offset_x:offset_x + n2] = img_above
        # 去除纵向衔接缝
        for i in range(offset_y, m2 + offset_y):
            for j in range(offset_x, n2 + offset_x):
                alpha = 1
                if not (np.array_equal(dst_img[i, j], np.array([0, 0, 0])) or j >= offset_x + blend_width):
                    alpha = (j - offset_x) / blend_width
                # if not (np.array_equal(dst_img[i, j], np.array([0, 0, 0])) or i >= offset_y + blend_height):
                #     alpha = (i - offset_y) / blend_height
                # 图像融合 (Blend)
                dst_img[i, j] = alpha * img_above[i - offset_y, j - offset_x] + (1 - alpha) * dst_img[i, j]
    else:
        # 融合宽度(★可调整·注意大小不能过大)
        blend_width = 100
        # 先放置above图像，发现衔接处融合效果会更好
        dst_img[0:m2, 0:n2] = img_above
        # 融合起始位置
        blend_start_x = n2 - blend_width
        # 不能简单的覆盖，而是进行互补操作，填补空缺的区域
        for i in range(m1):
            for j in range(n1):
                if i < m2 and j < n2:
                    # 融合条件
                    if j > blend_start_x:
                        if np.array_equal(dst_img[i, j], [0, 0, 0]):
                            alpha = 1
                        elif np.array_equal(img_below[i, j], [0, 0, 0]):
                            alpha = 0
                        else:
                            alpha = (j - blend_start_x) / blend_width
                        # 图像融合 (Blend)
                        dst_img[i, j] = alpha * img_below[i, j] + (1 - alpha) * dst_img[i, j]
                    else:
                        if np.array_equal(dst_img[i, j], [0, 0, 0]):
                            dst_img[i, j] = img_below[i, j]
                elif not np.array_equal(img_below[i, j], [0, 0, 0]):
                    dst_img[i, j] = img_below[i, j]
    # cv2.imshow('below', img_below)
    # cv2.imshow('above', img_above)
    # cv2.imshow('dst', dst_img)
    # cv2.waitKey()
    return dst_img


# 去除黑色边界
def remove_blank_edge(img):
    rows, cols = np.where(img[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    return img[min_row:max_row, min_col:max_col, :]


# 填补黑色区域
def padding_edge():
    pass


# 图像拼接（左图的像素点转换到右图对应位置）
@jit
def stitch_image_left_to_right(left, right):
    # 原始图像大小
    m_left, n_left, _ = np.shape(left)
    m_right, n_right, _ = np.shape(right)
    # 获取left和right的描述子
    kps1, des1 = extract_feature(left)
    kps2, des2 = extract_feature(right)
    # cv2.drawKeypoints(left, kps1, left, (255, 0, 0))
    # cv2.drawKeypoints(right, kps2, right, (255, 0, 0))
    # cv2.imshow('left', left)
    # cv2.imshow('right', right)
    # cv2.waitKey()
    # 获取最优匹配
    matches = get_nice_matches(des1, des2)
    # img_match = cv2.drawMatchesKnn(left, kps1, right, kps2, matches, None, flags=2)
    # cv2.imshow('matched', img_match)
    # cv2.waitKey()
    # 提取匹配点的位置坐标
    kps1_matched, kps2_matched = matches2points(kps1, kps2, matches)
    # 获取变换矩阵(★ 注意顺序！！！)
    matrix, mask = cv2.findHomography(kps2_matched, kps1_matched, cv2.RANSAC)
    # 计算最右侧边缘坐标
    matrix = np.linalg.inv(matrix)
    # 计算最左侧边缘坐标
    left_top_point = np.dot(matrix, np.array([0, 0, 1]))
    left_top_point = left_top_point / left_top_point[-1]
    left_down_point = np.dot(matrix, np.array([0, m_left, 1]))
    left_down_point = left_down_point / left_down_point[-1]
    # 变换矩阵-->平移
    matrix[0][-1] += abs(left_top_point[0])
    matrix[1][-1] += abs(left_top_point[1])
    # 偏移量
    offset_y = abs(int(left_top_point[1]))
    offset_x = abs(int(left_top_point[0]))
    # 边界宽度 (>0:/；<0\)
    edge_width = int(left_top_point[0] - left_down_point[0])
    # 计算最右侧边缘坐标
    right_point = np.dot(matrix, np.array([n_left, m_left, 1]))
    right_point = right_point / right_point[-1]
    # 拼接后的图像大小
    stitched_size = (int(right_point[0]) + offset_x, int(right_point[1]) + offset_y)
    # 执行坐标变换，将left上的点投影到right的对应位置
    left_transformed = cv2.warpPerspective(left, matrix, stitched_size)
    # For Test
    # left_transformed = cv2.line(left_transformed, (abs(int(left_top_point[0])), abs(int(left_top_point[1]))),
    #                             (abs(int(left_down_point[0])), abs(int(left_down_point[1]))), (255, 0, 0), 10)
    # cv2.imshow('1', left_transformed)
    # cv2.imshow('2', right)
    # cv2.waitKey()
    # 衔接处优化后的目标图像
    dst_img = optimize_seam(left_transformed, right, offset_x, offset_y, edge_width, stitch_type='left-right')
    return dst_img


# 图像拼接（右图的像素点转换到左图对应位置）
@jit
def stitch_image_right_to_left(left, right):
    # 原始图像大小
    m_left, n_left, _ = np.shape(left)
    m_right, n_right, _ = np.shape(right)
    # 获取img1和img2的描述子
    kps1, des1 = extract_feature(left)
    kps2, des2 = extract_feature(right)
    # 获取最优匹配
    matches = get_nice_matches(des1, des2)
    # img_match = cv2.drawMatchesKnn(left, kps1, right, kps2, matches, None, flags=2)
    # cv2.imshow('matched', img_match)
    # cv2.waitKey(0)
    # 提取匹配点的位置坐标
    kps1_matched, kps2_matched = matches2points(kps1, kps2, matches)
    # 获取变换矩阵(★ 注意顺序！！！)
    matrix, mask = cv2.findHomography(kps2_matched, kps1_matched, cv2.RANSAC)
    # 计算最右侧边缘坐标
    right_point = np.dot(matrix, np.array([n_right, m_right, 1]))
    right_point = right_point / right_point[-1]
    # 偏移量
    offset_y = int(right_point[1])
    offset_x = int(right_point[0])
    # 拼接后的图像大小
    stitched_size = (n_left + offset_x, m_left + offset_y)
    # 执行坐标变换，将right上的点投影到left的对应位置
    right_transformed = cv2.warpPerspective(right, matrix, stitched_size)
    # cv2.imshow('1', right_transformed)
    # cv2.imshow('2', left)
    # cv2.waitKey()
    # 左侧边缘坐标
    left_point = np.dot(matrix, np.array([0, 0, 1]))
    left_point = left_point / left_point[-1]
    offset_y = abs(int(left_point[1]))
    offset_x = abs(int(left_point[0]))
    # # 衔接处优化后的目标图像
    dst_img = optimize_seam(right_transformed, left, offset_x, offset_y, 0, stitch_type='right-left')
    return dst_img


if __name__ == '__main__':
    # 加载待拼接图像
    images = load_split_images()
    # 目标图像
    img_dst = None
    # 中间图片位置（默认前向缝合）
    center_index = len(images) - 1
    if STITCH_TYPE == 'center':
        center_index = int(len(images) / 2)
    elif STITCH_TYPE == 'forward':
        center_index = len(images) - 1
    elif STITCH_TYPE == 'backward':
        center_index = 0
    for img_index in range(0, len(images)):
        print('----->>> Stitching image {0} of {1}_{2}'.format(img_index, FOLDER_NAME, IMAGE_SET_NAME))
        if img_index == 0:
            img_dst = images[0]
        elif img_index <= center_index:
            # left->right
            print('left->right')
            img_dst = stitch_image_left_to_right(img_dst, images[img_index])
        else:
            # right->left
            print('right->left')
            img_dst = stitch_image_right_to_left(img_dst, images[img_index])
        # 去除黑色边界
        img_dst = remove_blank_edge(img_dst)
    output_path = './output/{0}_{1}.png'.format(FOLDER_NAME, IMAGE_SET_NAME)
    cv2.imwrite(output_path, img_dst)
