"""
Background vs Foreground Image segmentation. The goal is to produce a segmentation map that imitates
videocalls tools like the ones implemented in Google Meet, Zoom without using Deep Learning- or Machine Learning-
based techniques.  
   
This script does the following:  
- builds a background model using approximately the first 3s of the video, acting on the HSV colorspace;  
- performs frame differencing in the HSV domain;  
- runs LP filtering (median-filter) on the Saturation difference;  
- uses Otsu's technique to threshold the saturation and the brightness difference;  
- concatenates the saturation and the brightness masks to produce the foreground mask;  
- uses the foreground mask, the current video stream and a pre-defined background picture to produce the final output.  

Additionally, the script provides methods to reset the background (by pressing the 'r' key) and to save the current mask
(by pressing the 's' key). In the latter: 
- Lucas-Kanade optical flow algorithm is applied in order to continuously track the foreground points;
- Median LP filtering is performed to denoise the optical flow output;   
- the convex hull algorithm is applied to build a shape around the tracked points;   
- the resulting mask is then used with logical bitwise operators to separate foreground and background.   
   
Authors: M. Farina, F. Diprima - University of Trento
Last Update (dd/mm/yyyy): 03/04/2021 
"""

import os
import cv2
import time
import numpy as np
from helpers.variables import *
from helpers.utils import build_argparser, codec_from_ext, make_folder, recursive_clean


def run(**kwargs):
    """
        Main loop for background removal.
    """ 
    time_lst = [0]
    
    # setup an image for the background
    bg_pic_path = kwargs['background']
    bg_pic = cv2.imread(bg_pic_path)
    bg_pic = cv2.resize(bg_pic, dst_size)

    # setup the video writer if needed
    writer = None
    if kwargs["output_video"]:
        codec = codec_from_ext(kwargs["output_video"])
        writer = cv2.VideoWriter(kwargs["output_video"], codec, fps, frameSize=(width, height))

    # create the output frame folder if needed
    if kwargs["frame_folder"]:
        if kwargs["refresh"]: recursive_clean(kwargs["frame_folder"])
        make_folder(kwargs["frame_folder"])
    
    # initialize background
    hsv_bg = np.zeros(dst_shape_multi, dtype='uint16')
    black_bg = np.zeros(dst_shape_multi[:-1], dtype='uint8')

    # initialize vector of points for opticalFlow
    start_points = np.array([], dtype=np.float32)
    mask_saved = False
    prev_gray = None

    # start looping through frames
    frame_count = 0
    if cap.isOpened():
        while cap.isOpened():
            # retrieve the current frame and exit if needed
            ret, frame = cap.read()
            if not ret:
                break
            
            # otherwise, perform basic operations on the current frame
            frame = cv2.resize(frame, dst_size)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, ksize=gauss_kernel, sigmaX=2, sigmaY=2)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_frame_blurred = cv2.GaussianBlur(hsv_frame, gauss_kernel, sigmaX=2, sigmaY=2)
            
            # build a model for the background during the first frames
            if frame_count < bg_frame_limit:
                hsv_bg = hsv_bg.copy() + hsv_frame_blurred
                if frame_count == bg_frame_limit-1:
                    hsv_bg = np.uint8(hsv_bg.copy() / bg_frame_limit)

            # when the bg has been modeled, segment the fg
            else:
                time_in = time.perf_counter()

                # check if we should behave 'normally' or use opt-flow
                if mask_saved:

                    # if it is the first frame of the opt-flow method, initilize pts
                    # to be tracked ith the previous mask
                    if len(start_points) == 0:
                        indices = np.where(fg_mask_closed == 255)
                        start_points = np.array([[round(indices[1][i]), round(indices[0][i])] for i in range(len(indices[0]))], dtype=np.float32)
                        points = start_points
                        status = np.array([[1]]*len(points))
                    
                    # otherwise, run the Lucas Kanade algorithm
                    else:
                        prev_points = np.array(points, dtype=np.float32)
                        points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    
                    # retain only points successfully tracked
                    point_set = np.array(points, dtype=np.int32)
                    point_set_ = point_set.copy()[status[:,0]==1]
                    
                    # discard out of bounds indices
                    rows = point_set_[:, 1]
                    max_row_shift = max(rows) - n_rows + 1 if max(rows) >= n_rows else 1 
                    cols = point_set_[:, 0]
                    max_col_shift = max(cols) - n_cols + 1 if max(cols) >= n_cols else 1
                    rows = rows - max_row_shift
                    cols = cols - max_col_shift
                    
                    # build a convex hull around the tracked points and use it as a mask
                    fg_mask_closed = black_bg.copy()
                    fg_mask_closed[rows, cols] = 255
                    fg_mask_closed = cv2.medianBlur(fg_mask_closed.copy(), ksize=median_ksize)
                    point_set_median = np.where(fg_mask_closed == 255)
                    point_set_median = np.array([[round(point_set_median[1][i]), round(point_set_median[0][i])] for i in range(len(point_set_median[0]))], dtype=np.float32)                    
                    if len(point_set_median) > 0:
                        hull = np.array(cv2.convexHull(point_set_median), dtype=np.int32)
                        fg_mask_closed = cv2.fillConvexPoly(black_bg.copy(), hull, 255)
                    
                else:
                    # perform frame differencing
                    diff = cv2.absdiff(hsv_frame_blurred, hsv_bg)
                    h_diff, s_diff, v_diff = cv2.split(diff)
                    
                    # automatic global thresholding with Otsu's technique
                    r1, h_diff_thresh = cv2.threshold(h_diff, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    r2, s_diff_thresh = cv2.threshold(s_diff, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    r3, v_diff_thresh = cv2.threshold(v_diff, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    
                    # take into account contribution of saturation and value (aka 'brightness')
                    # clean the saturation mask beforehand, it usually is more unstable
                    s_diff_thresh_median = cv2.medianBlur(s_diff_thresh, ksize=median_ksize)
                    fg_mask = s_diff_thresh_median + v_diff_thresh
                    fg_mask_closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=10)
                    fg_mask_closed = cv2.dilate(fg_mask_closed.copy(), kernel)
                    
                # compute the actual foreground and background
                foreground = cv2.bitwise_and(frame, frame, mask=fg_mask_closed)
                background = bg_pic - cv2.bitwise_and(bg_pic, bg_pic, mask=fg_mask_closed)
                
                # ... and add them to generate the output image
                out = cv2.add(foreground, background)
                
                # display the output and the masks
                cv2.imshow("Output", out)
            
                # quit if needed
                key = cv2.waitKey(ms)
                if key==ord('q'):
                    break

                # if user presses 's' save and track the current mask with optical flow assumptions
                elif key == ord('s'):
                    mask_saved = True
                       
                # if user presses 'r', reset the background model
                elif key == ord('r'):
                    mask_saved = False
                    start_points = np.array([], dtype=np.float32)
                    frame_count = -1
                    hsv_bg = np.zeros(dst_shape_multi, dtype='uint16')
                
                # write the video on the fs if the user requested it
                if writer:
                    writer.write(cv2.resize(out, dsize=(width, height)))
                
                # save frames on the fs if the user requested it
                if kwargs["frame_folder"] and frame_count % kwargs["throttle"] == 0:
                    cv2.imwrite(os.path.join(kwargs["frame_folder"], "{}.jpg".format(frame_count - bg_frame_limit + 1)), out)

                # keep track of time
                time_out = time.perf_counter()
                time_diff = time_out - time_in
                time_lst.append(time_diff)

            prev_gray = gray_frame.copy()
            frame_count += 1

    print("Average Time x Frame: ", round(np.sum(np.array(time_lst))/len(time_lst), 2))
    cv2.destroyAllWindows()
    cap.release()
    if writer:
        writer.release()

if __name__ == "__main__":
    parser = build_argparser()
    kwargs = vars(parser.parse_args())
    run(**kwargs)
