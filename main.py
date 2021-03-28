import cv2
import sys
import numpy as np

# initialize video capture
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
ms = int(1000/fps)

# define the variables for image-processing tasks
kernel = np.ones((7,7), dtype='uint8')
n_rows = 480
n_cols = 720
dst_size = (n_cols, n_rows)
dst_shape = (n_rows, n_cols)
bg_frame_limit = fps * 3  # number of frames in 3 seconds

# setup an image to replace the background
bg_pic_path = "/home/teofa/Pictures/Background/ubuntu-wallpaper-hd.jpg"
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
            gray_frame_blurred = cv2.GaussianBlur(gray_frame, (5,5), sigmaX=2, sigmaY=2)
            
            # build a model for the background during the first frames
            if frame_count < bg_frame_limit:
                gray_bg = gray_bg.copy() + gray_frame_blurred
                if frame_count == bg_frame_limit-1:
                    gray_bg = np.uint8(gray_bg.copy() / bg_frame_limit)

            # when the bg has been modeled, segment the fg
            else:
                diff = cv2.absdiff(gray_frame_blurred, gray_bg)
                # threshold the difference to get the fg mask
                r, fg_mask = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)
                bg_mask = cv2.bitwise_not(fg_mask)
                # use the masks to compute fg and bg images
                fg = cv2.bitwise_and(frame, frame, mask=fg_mask)
                bg = cv2.bitwise_and(bg_pic, bg_pic, mask=bg_mask)
                out = fg + bg
                # display the output and the masks
                cv2.imshow("Background Model", gray_bg)
                cv2.imshow("Difference", diff)
                cv2.imshow("[Masks] Foreground vs Background", cv2.hconcat([fg_mask, bg_mask]))
                cv2.imshow("Output", out)
                # quit if needed
                if cv2.waitKey(ms) & 0xFF==ord('q'):
                    break
            
            frame_count += 1

    cv2.destroyAllWindows()
    cap.release()




if __name__ == "__main__":
    run()