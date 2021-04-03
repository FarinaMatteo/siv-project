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
Last Update (dd/mm/yyyy): 28/03/2021 
"""
import cv2
import numpy as np

# initialize video capture
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
ms = int(1000/fps)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# initialize video writer
codec = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('report_videos/main.mp4', codec, fps, frameSize=(width, height))

# define the variables for image-processing tasks
n_rows = 480
n_cols = 720
dst_size = (n_cols, n_rows)
dst_shape = (n_rows, n_cols)
bg_frame_limit = fps * 3  # number of frames in 3 seconds
gauss_kernel = (15,15)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3,5))

# setup an image to replace the background
bg_pic_path = "/home/teofa/Pictures/Background/catalina-day.jpg"
bg_pic = cv2.imread(bg_pic_path)
bg_pic = cv2.resize(bg_pic, dst_size)

def run():
    """
        Main loop for background removal. Uses frame differencing together with
        morphological operators to separate background from foreground for videoconferencing.
    """
    # initialize background
    gray_bg = np.zeros(dst_shape, dtype='uint16')
    
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
                diff = cv2.absdiff(gray_frame_blurred, gray_bg)
                
                # threshold the difference to get the fg mask
                r_otsu, fg_mask_otsu = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # perform low-pass and morphological operations on the masks
                fg_mask_otsu_closed = cv2.morphologyEx(fg_mask_otsu, cv2.MORPH_CLOSE, kernel=kernel, iterations=3)
                fg_mask_otsu_eroded = cv2.erode(fg_mask_otsu_closed, kernel=kernel)
                
                # use the masks to invert them and build the bg ones
                bg_mask_otsu = cv2.bitwise_not(fg_mask_otsu_eroded)
                
                # use the masks to compute fg and bg images
                fg_otsu = cv2.bitwise_and(frame, frame, mask=fg_mask_otsu_eroded)
                bg_otsu = cv2.bitwise_and(bg_pic, bg_pic, mask=bg_mask_otsu)
                out_otsu = fg_otsu + bg_otsu
                
                # display the output and the masks
                cv2.imshow("Output", out_otsu)
                # quit if needed
                if cv2.waitKey(ms) & 0xFF==ord('q'):
                    break

                if writer:
                    writer.write(cv2.resize(out_otsu, dsize=(width, height)))
                
            
            frame_count += 1

    cv2.destroyAllWindows()
    cap.release()
    if writer:
        writer.release()




if __name__ == "__main__":
    run()