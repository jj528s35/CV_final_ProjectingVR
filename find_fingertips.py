import numpy as np
import math
import random
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tracking

def find_fingertip(hand_mask, debug = False):
    '''
    return: 
    @cnt: contours
    Smooth the mask and find the contours
    convert the result image into RGB image ==> only for visual
    Find the contour with max area ==> contour of the hand mask
    get centroid from hand, then Draw the center and Hull_Convex
    Find the Fingertips by travel the contour
    '''
    fingertips = []
    kernel_size = 7
    
    #Smooth the mask and find the contours
    smooth_mask = cv2.GaussianBlur(hand_mask,(kernel_size, kernel_size), 0)
    (_, cnts, _) = cv2.findContours(smooth_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #convert the result image into RGB image
    contours_image = cv2.convertScaleAbs(smooth_mask, alpha=(255))
    contours_image = cv2.cvtColor(contours_image, cv2.COLOR_GRAY2BGR)
    
    # Find the contour with max area
    maxArea = 0
    hull = None
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        if area > maxArea:
            maxArea = area
            hull = cv2.convexHull(cnts[i])
            Contours = cnts[i]
    
    cx, cy = (0,0)
    if hull is not None : 
        # get centroid from hand, then Draw the center and Hull_Convex
        M = cv2.moments(hull)
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        if(debug == True):
            cv2.circle(contours_image, (cx, cy), 5, (0,0,255), 3)
        cv2.drawContours(contours_image, [Contours], -1, (255,0,0), 2)
        
        #Find the Fingertips
#         start = time.time()
        fingertips, contours_image = find_fingertips_by_dot(Contours, hull, contours_image,(cx, cy), debug)
#         print('finger: %.4f'%(time.time() - start))

    return cnts, contours_image, (cx, cy), np.asarray(fingertips, dtype=np.int32)


def find_fingertips_by_dot(Contours, hull, contours_image, center, debug=False):
    """
    if the point is in the convex list and there are not points near to it haved be labeled, then add it to Fingertip list
    If its near points labeled, then choose the points with larger curvature (dot value is smaller)
    """
    fingertips = []
    skip = 4 # 4 ==> 點少但無誤判
    Convex = (0,0)
    post_dot = 20
    threshold = 8
    Convex_threshold = 1
    Dot_threshold = 20
    first = True
    for i in range(skip, len(Contours)-skip):
        p = Contours[i-skip]
        q = Contours[i]
        r = Contours[i+skip]

        dot = np.dot(p-q,(r-q).T)
        if (dot < Dot_threshold and dot > -Dot_threshold):
            points = (q[0,0],q[0,1])
            if first == True:
                Convex = points
                s_point = 0
            IsnotEdge = points[0]!=0 and points[1]!=0 and points[0]!=contours_image.shape[1] and points[1]!=contours_image.shape[0]
            IsConvex, s_point = Points_is_convex(hull, points, Convex_threshold, s_point)
            Near_Labeled = abs(Convex[0] - points[0]) > threshold and abs(Convex[1] - points[1]) > threshold
            #if the point is in the convex list and haven't be labeled, then add it to Fingertip list
            if IsConvex and IsnotEdge:
                if Near_Labeled or first:
                    fingertips.append(points)
                    first = False
                elif dot < post_dot:#若在附近，則取曲率較大的點
                    fingertips[-1] = points
                Convex = points
                post_dot = dot
                    
    if(debug==True):
        for i in range(len(fingertips)):
            cv2.circle(contours_image, fingertips[i], 5 , (255,255,0) , 3)

    return fingertips, contours_image

def Points_is_convex(hull, points, threshold, s_point):
    """
    check whether the contour points is in convex points list, and there have a acceptable error range(threshold)
    and start at s_point, don't need to travel all the contour points
    """
    label = False
    label_hull = hull.copy()
    for i in range(s_point, len(hull)):
        if abs(hull[i,0,0] - points[0]) <= threshold and abs(hull[i,0,1] - points[1]) <= threshold:
            label = True
            if s_point - 5 >= 0:
                s_point = i - 5
            return label, s_point
    return label, s_point


