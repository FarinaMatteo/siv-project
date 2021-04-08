"""
Background vs Foreground Image segmentation. The goal is to produce a segmentation map that imitates
videocalls tools like the ones implemented in Google Meet, Zoom without using Deep Learning- or Machine Learning-
based techniques.  
This script does the following:  
- builds a background model using the first 3s of the video, acting on the HSV colorspace;  
- performs frame differencing in the HSV domain;  
- runs LP filtering (median-filter) on the Saturation difference;  
- uses Otsu's technique to threshold the saturation and the brightness difference;  
- concatenates the saturation and the brightness masks to produce the foreground mask;  
- runs morphological operators one the mask (closing and dilation) with a 3x5 ellipse (resembles the shape of a human face);  
- uses the foreground mask, the current video stream and a pre-defined background picture to produce the final output.  
  
Authors: M. Farina, F. Diprima - University of Trento
Last Update (dd/mm/yyyy): 09/04/2021 
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
                fg_mask_dilated = cv2.dilate(fg_mask_closed, kernel=kernel, iterations=3)
                
                # compute the actual foreground and background
                foreground = cv2.bitwise_and(frame, frame, mask=fg_mask_dilated)
                background = bg_pic - cv2.bitwise_and(bg_pic, bg_pic, mask=fg_mask_dilated)
                
                # ... and add them to generate the output image
                out = cv2.add(foreground, background)
                
                # display the output and the masks
                cv2.imshow("Output", out)
        
                # save frames on the fs if the user requested it
                if kwargs["frame_folder"] and frame_count % kwargs["throttle"] == 0:
                    cv2.imwrite(os.path.join(kwargs["frame_folder"], "{}.jpg".format(frame_count - bg_frame_limit + 1)), out)

                # write the video on the fs if the user requested it
                if writer:
                    writer.write(cv2.resize(out, dsize=(width, height)))
                
                # quit if needed
                if cv2.waitKey(ms) & 0xFF==ord('q'):
                    break
                
                # keep track of time
                time_out = time.perf_counter()
                time_diff = time_out - time_in
                time_lst.append(time_diff)

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