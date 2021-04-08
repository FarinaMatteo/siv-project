import cv2

# initialize video capture
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
ms = int(1000/fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# define the variables for image-processing tasks
n_rows = 480
n_cols = 720
dst_size = (n_cols, n_rows)
dst_shape_gray = (n_rows, n_cols)
dst_shape_multi = (n_rows, n_cols, 3)
bg_frame_limit = fps * 3  # number of frames in 3 seconds
gauss_kernel = (25,25)
median_ksize = 15
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3,5))
