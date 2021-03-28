"""
Background vs Foreground Image segmentation. The goal is to produce a segmentation map that imitates
videocalls tools like the ones implemented in Google Meet, Zoom without using Deep Learning- or Machine Learning-
based techniques.  
This script does the following:  
- builds a background model using the first 3s of the video, acting on the HSV colorspace;  
- performs frame differencing in the HSV domain;  
- runs LP filtering (median-filter) on the Saturation difference;  
- interactively displays bars for Saturation and Brightness thresholds;  
- performs image binary thresholding on the Saturation and Brightness channels using the user-defined threshold values;  
- concatenates the saturation and the brightness masks to produce the foreground mask;  
- runs morphological operators one the mask (closing and erosion) with a 3x5 ellipse (resembles the shape of a human face);  
- uses the mask, the current video stream and a pre-defined background picture to produce the final output.  
  
Authors: M. Farina, F. Diprima - University of Trento
Last Update (dd/mm/yyyy): 28/03/2021 
"""
import cv2
import time
import numpy as np
from random import randint

# initialize video capture
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
ms = int(1000/fps)

# define the variables for image-processing tasks
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3,5))
n_rows = 480
n_cols = 720
dst_size = (n_cols, n_rows)
dst_shape = (n_rows, n_cols, 3)
sat_thresh = randint(0, 255)
val_thresh = randint(0, 255)
bg_frame_limit = fps * 3  # number of frames in 3 seconds
gauss_kernel = (15,15)
median_ksize = 11

# setup an image to replace the background
bg_pic_path = "/home/teofa/Pictures/Background/tux_wallpaper.jpg"
bg_pic = cv2.imread(bg_pic_path)
bg_pic = cv2.resize(bg_pic, dst_size)

# define callbacks for trackbars
def on_sat_tb(value):
    global sat_thresh
    sat_thresh = value

def on_val_tb(value):
    global val_thresh
    val_thresh = value

# main function
def run():
    """
        Main loop for background removal. Uses frame differencing together with
        morphological operators to separate background from foreground for videoconferencing.
    """
    time_lst = [0]
    
    # initialize background
    hsv_bg = np.zeros(dst_shape, dtype='uint16')

    # initialize windows and trackbars
    cv2.namedWindow("Output - Weighted vs Original")
    cv2.createTrackbar("saturation threshold", "Output - Weighted vs Original", sat_thresh, 255, on_sat_tb)
    cv2.createTrackbar("brightness threshold", "Output - Weighted vs Original", val_thresh, 255, on_val_tb)

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
                # perform thresholding with the user defined values
                r2, s_diff_thresh = cv2.threshold(s_diff, sat_thresh, 255, cv2.THRESH_BINARY)
                r3, v_diff_thresh = cv2.threshold(v_diff, val_thresh, 255, cv2.THRESH_BINARY)
                # take into account contribution of saturation and value (aka 'brightness')
                # clean the saturation mask beforehand, it usually is more unstable
                s_diff_thresh_median = cv2.medianBlur(s_diff_thresh, ksize=median_ksize)
                fg_mask = s_diff_thresh_median + v_diff_thresh
                fg_mask_closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=3)
                fg_mask_eroded = cv2.erode(fg_mask_closed, kernel=kernel, iterations=1)
                # compute the actual foreground and background
                foreground = cv2.bitwise_and(frame, frame, mask=fg_mask_eroded)
                background = bg_pic - cv2.bitwise_and(bg_pic, bg_pic, mask=fg_mask_eroded)
                # ... and add them to generate the output image
                output_weighted = cv2.addWeighted(foreground, 0.8, bg_pic, 0.2, 0)
                out = cv2.add(foreground, background)
                # display the output and the masks
                cv2.imshow("Output - Weighted vs Original", cv2.hconcat([output_weighted, out]))
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




if __name__ == "__main__":
    run()