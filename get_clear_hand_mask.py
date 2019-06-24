import numpy as np
import math
import random
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tracking


# # RANSAM

def Dis_pt2plane(pts, a, b, c, d):
    """
    Compute the distance from points to the plane
    """
    normal = math.sqrt(a*a+b*b+c*c)
    if normal == 0:
        normal = 1
    
    v = np.array([a,b,c])
    dis = abs(np.dot(pts,v.T)+d)/normal
    return dis

def get_Plane(sampts):
    """
    Compute the equation of the plane
    """
    p1 = sampts[0]
    p2 = sampts[1]
    p3 = sampts[2]
    
    a = ( (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]) )
    b = ( (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]) )
    c = ( (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]) )
    d = ( 0-(a*p1[0]+b*p1[1]+c*p1[2]) )
    
    return a,b,c,d

def Random3points(points3D, ConfidenceIndex):
    """
    Random choose 3 Confidence points
    """
    sample_number = 3
    sample_point_index = random.sample(range(ConfidenceIndex.shape[0]), sample_number)
    sample_points = np.zeros((sample_number,3))
    for i in range(sample_number):
        Confidence_point_index = sample_point_index[i]
        index = ConfidenceIndex[Confidence_point_index]
        y = index // points3D.shape[1]
        x = index % points3D.shape[1]
        sample_points[i] = points3D[y][x]
    return sample_points

# def Random3points(points3D):
#     sample_number = 3
#     sample_point_index = random.sample(range(points3D.shape[0]*points3D.shape[1]), sample_number)
#     sample_points = np.zeros((sample_number,3))
#     for i in range(sample_number):
#         index = sample_point_index[i]
#         y = index // points3D.shape[1]
#         x = index % points3D.shape[1]
#         sample_points[i] = points3D[y][x]
#     return sample_points

def get_inliner_num(points3D,a,b,c,d,inliner_threshold):
    """
    Compute the liner points which distance to plane < threshold
    Also get distance from points to the plane (new Depth Image which re-project depth pixels in surface plane)
    """
    inliner_num = 0
    
    dist = Dis_pt2plane(points3D,a,b,c,d)
    inliner_mask = dist < inliner_threshold
    inliner_num = np.sum(inliner_mask)
    return inliner_num, inliner_mask, dist

def RANSAM(points3D, ConfidenceIndex, ransac_iteration = 1000, inliner_threshold = 0.01):
    best_inlinernum = -1
    best_inlinernum = 0
    best_plane = np.zeros((1,4))
    best_depthImage = np.zeros((points3D.shape[0],points3D.shape[1]))
    best_plane_mask = np.zeros((points3D.shape[0],points3D.shape[1]))
#     best_sampts = np.zeros((3,3))
    
#     print(points3D.shape,points3D[80:90,110])
    for i in range(ransac_iteration):
        sampts = Random3points(points3D, ConfidenceIndex)
        a,b,c,d = get_Plane(sampts)
        
        inliner_num, inliner_mask, depthImage = get_inliner_num(points3D,a,b,c,d,inliner_threshold)
        if(inliner_num > best_inlinernum):
            best_inlinernum = inliner_num
            best_plane = np.array([a,b,c,d])
            best_plane_mask = inliner_mask
            best_depthImage = depthImage
#             best_sampts = sampts
            
    print("Inliner Number\n", best_inlinernum)
    print("Inliner plane\n", best_plane)
    return best_plane, best_depthImage, best_plane_mask


# # Depth Map

def get_depth_map(points3D,plane):
    """
    Compute the liner points which distance to plane < threshold
    Also get distance from points to the plane (new Depth Image which re-project depth pixels in surface plane)
    """
    dist = Dis_pt2plane(points3D,plane[0],plane[1],plane[2],plane[3])
    return dist


# # Edge map

def get_edge_map(grayImage,depthImage):
    """
    Canny Edge map
    turn grayImg from int32 to int8
    blur the grayImg then do Canny Edge
    """
    low_threshold = 2
    high_threshold = 10
    grayimg_int8 = cv2.convertScaleAbs(grayImage, alpha=(255.0/65535.0))
    
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(grayimg_int8,(kernel_size, kernel_size), 0)
    Cannyedges = cv2.Canny(grayimg_int8, low_threshold, high_threshold)#blur_gray
    
    """
    Threshold based Edge map
    if depth between the pixel and its nearby pixels > near_depth_threshold, then labeled it
    """
    s_time = time.time()
    near_depth_threshold = 0.05 #0.05
#     print(np.max(depthImage))
    Threshold_based_edge = np.zeros((depthImage.shape[0],depthImage.shape[1]))
    
    h = depthImage.shape[0]
    w = depthImage.shape[1]
    depth_img_transform = np.zeros((h+1,w+1))
    depth_img_transform[:h,:w] = depthImage
    #check left up depth threshold
    depth_img_transform[1:h+1,1:w+1] = depthImage
    check_depth_threshold = abs(depthImage - depth_img_transform[:h,:w]) > near_depth_threshold
    Threshold_based_edge = np.logical_or(Threshold_based_edge, check_depth_threshold)
    #check up depth threshold
    depth_img_transform[:h,:w] = depthImage
    depth_img_transform[1:h+1,:w] = depthImage
    check_depth_threshold = abs(depthImage - depth_img_transform[:depthImage.shape[0],:depthImage.shape[1]]) > near_depth_threshold
    Threshold_based_edge = np.logical_or(Threshold_based_edge, check_depth_threshold)
    #check Right up depth threshold
    depth_img_transform[:h,:w] = depthImage
    depth_img_transform[1:h+1,:w-1] = depthImage[:,1:w]
    check_depth_threshold = abs(depthImage - depth_img_transform[:h,:w]) > near_depth_threshold
    Threshold_based_edge = np.logical_or(Threshold_based_edge, check_depth_threshold)
    #check Left depth threshold
    depth_img_transform[:h,:w] = depthImage
    depth_img_transform[:h,1:w+1] = depthImage
    check_depth_threshold = abs(depthImage - depth_img_transform[:h,:w]) > near_depth_threshold
    Threshold_based_edge = np.logical_or(Threshold_based_edge, check_depth_threshold)
    #check Right depth threshold
    depth_img_transform[:h,:w] = depthImage
    depth_img_transform[:h,:w-1] = depthImage[:,1:w]
    check_depth_threshold = abs(depthImage - depth_img_transform[:h,:w]) > near_depth_threshold
    Threshold_based_edge = np.logical_or(Threshold_based_edge, check_depth_threshold)
    #check Left down depth threshold
    depth_img_transform[:h,:w] = depthImage
    depth_img_transform[:h-1,1:w+1] = depthImage[1:h,:]
    check_depth_threshold = abs(depthImage - depth_img_transform[:h,:w]) > near_depth_threshold
    Threshold_based_edge = np.logical_or(Threshold_based_edge, check_depth_threshold)
    #check down depth threshold
    depth_img_transform[:h,:w] = depthImage
    depth_img_transform[:h-1,:w] = depthImage[1:h,:]
    check_depth_threshold = abs(depthImage - depth_img_transform[:h,:w]) > near_depth_threshold
    Threshold_based_edge = np.logical_or(Threshold_based_edge, check_depth_threshold)
    #check Right down depth threshold
    depth_img_transform[:h,:w] = depthImage
    depth_img_transform[:h-1,:w-1] = depthImage[1:h,1:w]
    check_depth_threshold = abs(depthImage - depth_img_transform[:h,:w]) > near_depth_threshold
    Threshold_based_edge = np.logical_or(Threshold_based_edge, check_depth_threshold)
    
    
#     print('*get threshold edge: %.4f s'%(time.time()-s_time))
    """
    Merge Canny Edge map and Threshold based Edge map
    """
    Edge_map = np.logical_or(Cannyedges,Threshold_based_edge)
    
    return Cannyedges,Threshold_based_edge, Edge_map, blur_gray


# # Find High Region


def get_high_region(depthImage):
    """
    Define plane edge
    High region : > 0.04 
    ==> Flood fill mask : if pixels value > 0.04, then its value of mask image = 0 
    ==> mask = depthImage < 0.04
    
    Next, reject the region which dose not connet with plane edge and record its position
    ==>Just do Flood fill around the plane edge
    """
    h, w = depthImage.shape[:2]
    high_region_mask = np.ones((h+2,w+2), np.uint8)
    high_region_mask[1:h+1,1:w+1] = depthImage < 0.04 # > 0.04 False == 0, Flood fill will fill pixels with 0
    resultImg = np.zeros((h,w), np.uint8)
    
    maxArea = 0
    max_high_pos = (-1,-1)
    
    #define plane edge
    plane_edge = 15#25
    
    x1 = plane_edge
    x2 = w - plane_edge
    y1 = plane_edge
    y2 = h - plane_edge
    
    high_list = []
    
    for y in range(plane_edge, y2):
        if high_region_mask[y+1,x1+1] == 0 and resultImg[y,x1] != True:
            cv2.floodFill(resultImg, high_region_mask.copy(), (x1, y),True, cv2.FLOODFILL_MASK_ONLY)
#             high_list.append((x1,y))
            maxArea, max_high_pos, area = max_area_pos(high_region_mask, (x1, y), maxArea, max_high_pos)
#             ind = get_max_ind(depthImage, high_region_mask.copy(), x1, y)
#             if area == maxArea: 
#                 max_high_pos = ind

        if high_region_mask[y+1,x2+1] == 0 and resultImg[y,x2] != True:
            cv2.floodFill(resultImg, high_region_mask.copy(), (x2, y),True, cv2.FLOODFILL_MASK_ONLY)
#             high_list.append((x2,y))
            maxArea, max_high_pos, area = max_area_pos(high_region_mask, (x2, y), maxArea, max_high_pos)
#             ind = get_max_ind(depthImage, high_region_mask, x2, y)
#             if area == maxArea: 
#                 max_high_pos = ind#high_list.append(ind)
            
            
    for x in range(plane_edge, x2):
        if high_region_mask[y1+1,x+1] == 0 and resultImg[y1,x] != True:
            cv2.floodFill(resultImg, high_region_mask.copy(), (x, y1),True, cv2.FLOODFILL_MASK_ONLY)
#             high_list.append((x, y1))
            maxArea, max_high_pos, area = max_area_pos(high_region_mask, (x, y1), maxArea, max_high_pos)
#             ind = get_max_ind(depthImage, high_region_mask, x, y1)
#             if area == maxArea: 
#                 max_high_pos = ind

        if high_region_mask[y2+1,x+1] == 0 and resultImg[y2,x] != True:
            cv2.floodFill(resultImg, high_region_mask.copy(), (x, y2),True, cv2.FLOODFILL_MASK_ONLY)
#             high_list.append((x, y2))
            maxArea, max_high_pos, area = max_area_pos(high_region_mask, (x, y2), maxArea, max_high_pos)
#             ind = get_max_ind(depthImage, high_region_mask, x, y2)
#             if area == maxArea: 
#                 max_high_pos = ind

            
    if max_high_pos != (-1,-1) and maxArea > 300:
        high_list.append(max_high_pos)
        print("pos : ",max_high_pos)
   
    return resultImg, high_list


def max_area_pos(high_region_mask, curr_pos, maxArea, max_high_pos, Debug = False):
    h, w = high_region_mask.shape[:2]
    h = h - 2
    w = w - 2
    mask = np.zeros((h,w), np.uint8)
    cv2.floodFill(mask, high_region_mask.copy(), curr_pos ,True, cv2.FLOODFILL_MASK_ONLY)
    
    localmax_area = 0
    hull = None
    (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        if area > localmax_area:
            localmax_area = area
            hull = cv2.convexHull(cnts[i])
            
    cx, cy = (0,0)
    if hull is not None and localmax_area >= maxArea:
        maxArea = localmax_area
        # get centroid from hand, then Draw the center and Hull_Convex
        M = cv2.moments(hull)
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        if(mask[cy,cx] == True):
            max_high_pos = (cx,cy)
        else:
            max_high_pos = curr_pos
    
        if Debug:
            mask = mask.copy()
            image = cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(image, [hull], -1, (255,255,0), 2)
            cv2.circle(image, max_high_pos, 5 , (0,0,255) , 3)
            plt.imshow(image)
            plt.show()
            plt.draw()
        
    return maxArea, max_high_pos, localmax_area


# # Hand Mask


def get_Hand_mask(Edge_map, high_list, High_region_Image, depth_only_mask = False):
    """
    Flood fill from high region position and stop when reach edge
    Only fill in the hight region mask 
    ==> Prevent flood fill from seed which filled region does not the hight region
    """
    h, w = Edge_map.shape[:2]
    
    resultImg = Edge_map.copy()
    resultImg.dtype = 'uint8'
    mask = np.zeros((h+2,w+2), np.uint8)
    mask1 = np.ones((h+2,w+2), np.uint8)
    mask1[1:h+1,1:w+1] = High_region_Image == False
    
    # depth-only tracking approach
    if depth_only_mask:
        mask = mask1

    for i in range(len(high_list)):
        cv2.floodFill(resultImg, mask, high_list[i],True,cv2.FLOODFILL_FIXED_RANGE)
    
    resultImg = resultImg - Edge_map
    
    return resultImg

def get_reasonable_Hand_mask(Edge_map, high_region_list, High_region_Image, points3D):
    Hand_mask_Image = get_Hand_mask(Edge_map, high_region_list, High_region_Image)
    #check wheather Flood_fill_with_edge_reasonable
    reasonable = Flood_fill_with_edge_reasonable(Hand_mask_Image, high_region_list, points3D)
    if reasonable == False:
        #Get Hand mask by Flood fill from high region position with Edge map
        Hand_mask_Image = get_Hand_mask(Edge_map, high_region_list, High_region_Image, depth_only_mask = True)
    return Hand_mask_Image


def Flood_fill_with_edge_reasonable(hand_mask_Img, high_list, points3D):
    """
    If the flood fill doesn’t stop (filling farther than a reasonable human finger length, 15 cm)
    If distance between hight region center and contour of flood fill region > 15
    then return False and roll back the hand mask to depth only(hight region)
    """
    reasonable = True
    (_, cnts, _) = cv2.findContours(hand_mask_Img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 0:
        # Find the contour with max area
        maxArea = 0
        hull = None
        Contours = cnts[0]
        for i in range(len(cnts)):
            area = cv2.contourArea(cnts[i])
            if area > maxArea:
                maxArea = area
                hull = cv2.convexHull(cnts[i])
                Contours = cnts[i]

        for i in range(len(Contours)):
            q = Contours[i]
            points = (q[0,0],q[0,1])
            dist = tracking.get_euclidean_distance(high_list[0],points, points3D)
            if dist > 0.15:
                reasonable = False
                print("Dist: ",dist)
                break
            
    return reasonable
    

