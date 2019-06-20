import numpy as np
import cv2

hands_center = np.zeros((2, 2))
fingertips = np.zeros((10,2))

def get_euclidean_distance(x, y, threed_points):
    a = threed_points[int(x[1]), int(x[0])]
    b = threed_points[int(y[1]), int(y[0])]
    return np.power((np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2) + np.power(a[2] - b[2], 2)), 0.5)
   
def get_tracking_index(tracking_array, new_tracking_positions, threed_points, euclidean_distance = 0.1):
    '''
        return:
        found_index[x, 0] = index
        found_index[x, 1] = distance
    '''
    tracking_size, _ = tracking_array.shape
    if(len(new_tracking_positions)>0):
        new_position_size, _= new_tracking_positions.shape
        pairs = np.zeros((new_position_size, 2))
    else:
        new_position_size = 0
        pairs = []
        
    # track the closet point
    for j in range(1, new_position_size):
        index = -1
        distance = -1
        min_distance = euclidean_distance
        for i in range(1, tracking_size):
            # skip no information
            if((tracking_array[i][0] == -1) and (tracking_array[i][1]==-1)):
                continue
            distance = get_euclidean_distance(tracking_array[i], new_tracking_positions[j], threed_points)
            if(distance < min_distance):
                index = i
                min_distance = distance
        pairs[j][0] = index
        pairs[j][1] = min_distance
#     print('pairs: ', pairs)
        
    # remove no udpate position
    for i, h in enumerate(tracking_array):
        need_udate = False
        for j, p in enumerate(pairs):
            if(pairs[j, 0] == i):
                need_udate = True
                # update new postion
                tracking_array[i] = new_tracking_positions[j]
                break
        if(need_udate == False):
            # remove no udate position
            tracking_array[i] = [-1, -1]
    
    # insert new position
    for i, p in enumerate(pairs):
        if(pairs[i][0] == -1):
            for j, item in enumerate(tracking_array):
                if((tracking_array[j][0] == -1) and (tracking_array[j][1] == -1)):
                    tracking_array[j] = new_tracking_positions[i]
                    break
        
    return tracking_array

def hand_tracking(new_center, new_tips, threed_points, draw_image=None):
    '''
        by Yuan-Syun Ye
        
        Hand from the current frame are matched to hands in the previous frame by Euclidean distance with a fixed upper limit
        on movement (i.e., assuming hands do not move more than 10 cm in a single, or 2.50 m/s).
    '''
    global hands_center, fingertips
    
    # turple to array
    new_center = np.array([[new_center[0], new_center[1]]])
    
#     print('-------------Hand Tracking---------------')
    # using Euclidean distance to tracking each postion.
    hands_center = get_tracking_index(hands_center, new_center, threed_points)
    fingertips = get_tracking_index(fingertips, new_tips, threed_points)
    
#     print('tracking hand center: ', hands_center)
#     print('tracking fingertips: ', fingertips)
#     print('----------------------------')

    if(draw_image is not None):
        tracking_image = draw_image.copy()
        hand_color = (0, 255, 0)
        fingertip_color = (0, 255, 0)
        text_size = 0.25
        for i, h in enumerate(hands_center):
            if(h[0] != -1):
                hand_text = 'h_' + str(i)
                pos = (int(hands_center[i][0]), int(hands_center[i][1]))
                cv2.putText(img=tracking_image, text=hand_text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=hand_color)
        for j, f in enumerate(fingertips):
            if(f[0] != -1):
                finger_text = 'f_' + str(j)
                pos = (int(fingertips[j][0]), int(fingertips[j][1]))
                cv2.putText(img=tracking_image, text=finger_text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=fingertip_color)
                
    return hands_center, fingertips, tracking_image