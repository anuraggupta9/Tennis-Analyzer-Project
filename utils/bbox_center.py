import math

def get_center(bbox):
    x1,y1,x2,y2 = bbox
    return (x1+x2)//2, (y1+y2)//2

def measure_distance(p1, p2):
    
    return math.sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)

def get_foot_positions(bbox):
    x1, y1, x2, y2 = bbox
    
    return ((x1+x2)/2, y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
   closest_distance = float('inf')
   key_point_ind = keypoint_indices[0]
   for keypoint_indix in keypoint_indices:
       keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
       distance = abs(point[1]-keypoint[1])

       if distance<closest_distance:
           closest_distance = distance
           key_point_ind = keypoint_indix
    
   return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])
