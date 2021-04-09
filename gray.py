"""
Background vs Foreground Image segmentation. The goal is to produce a segmentation map that imitates
videocalls tools like the ones implemented in Google Meet, Zoom without using Deep Learning- or Machine Learning-
based techniques.  
This script does the following:  
- builds a background model using the first 3s of the video stream;  
- performs frame differencing in the gray-scale domain;  
- uses Otsu's technique for global thresholding to produce the foreground and background masks;  
- uses the masks, the current video stream and a pre-defined background picture to produce the final output.  

The result is clearly unsatisfactory. Check 'hsv.py', 'trackbars.py' and 'hsv_optflow.py' for significant improvements.  

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
    gray_bg = np.zeros(dst_shape_gray, dtype='uint16')
    
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
            gray_frame_blurred = cv2.GaussianBlur(gray_frame, gauss_kernel, sigmaX=2, sigmaY=2)
            
            # build a model for the background during the first frames
            if frame_count < bg_frame_limit:
                gray_bg = gray_bg.copy() + gray_frame_blurred
                if frame_count == bg_frame_limit-1:
                    gray_bg = np.uint8(gray_bg.copy() / bg_frame_limit)

            # when the bg has been modeled, segment the fg
            else:
                time_in = time.perf_counter()
                diff = cv2.absdiff(gray_frame_blurred, gray_bg)
                
                # threshold the difference to get the fg mask
                r_otsu, fg_mask_otsu = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # perform low-pass and morphological operations on the masks
                fg_mask_otsu_closed = cv2.morphologyEx(fg_mask_otsu, cv2.MORPH_CLOSE, kernel=kernel, iterations=10)
                fg_mask_otsu_dilated = cv2.dilate(fg_mask_otsu_closed, kernel=kernel)
                
                # use the masks to invert them and build the bg ones
                bg_mask_otsu = cv2.bitwise_not(fg_mask_otsu_dilated)
                
                # use the masks to compute fg and bg images
                fg_otsu = cv2.bitwise_and(frame, frame, mask=fg_mask_otsu_dilated)
                bg_otsu = cv2.bitwise_and(bg_pic, bg_pic, mask=bg_mask_otsu)
                out_otsu = fg_otsu + bg_otsu
                
                # display the output and the masks
                cv2.imshow("Output", out_otsu)

                # write frames if the user requested it
                if kwargs["frame_folder"] and frame_count % kwargs["throttle"] == 0:
                    cv2.imwrite(os.path.join(kwargs["frame_folder"], "{}.jpg".format(frame_count - bg_frame_limit + 1)), out_otsu)

                # append the current frame to the output video if the user requested it
                if writer:
                    writer.write(cv2.resize(out_otsu, dsize=(width, height)))
                
                # quit if needed
                if cv2.waitKey(ms) & 0xFF==ord('q'):
                    break

                # keep track of time
                time_out = time.perf_counter()
                time_diff = time_out - time_in
                time_lst.append(time_diff)
                
            
            frame_count += 1

    # clean resource usage
    print("Average Time x Frame: ", round(np.sum(np.array(time_lst))/len(time_lst), 2))
    cv2.destroyAllWindows()
    cap.release()
    if writer:
        writer.release()


if __name__ == "__main__":
    parser = build_argparser()
    kwargs = vars(parser.parse_args())
    run(**kwargs)