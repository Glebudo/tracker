import cv2

def get_frame_size(cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_width, frame_height

def set_frame_size(cap):
    pass