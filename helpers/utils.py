"""
Utility functions to be imported in each script of the project.
"""

def build_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-bg", "--background", type=str, default='sample_imgs/catalina-day.jpg', required=False,
                        help='Path to the image used as background. Default=\'sample_imgs/catalina-day.jpg\' \
                            If relative, please provide it with respect to the python script you are running. Absolute path is preferred.')
    
    parser.add_argument("-ff", "--frame_folder", type=str, required=False,
                        help="Folder where to save processed frames. No frames will be saved if this argument is not given. \
                            It must be relative to the location of the script that you're running.")
    
    parser.add_argument('-r', '--refresh', required=False, action='store_true', 
                        help="Add this flag if you want to delete the content inside 'frame_folder' before saving the frames \
                            of this new run.")
    
    parser.add_argument('-t', '--throttle', required=False, default=1, type=int,
                        help="Skip int(throttle) frames when saving on disk. Default to 1 (no frames skipped).")
    
    parser.add_argument("-ov", "--output_video", type=str, required=False, 
                        help="Path to the output video generated concatenating processed frames. \
                            No output video will be generated if this argument is not given. \
                            It must be relative to the location of the script that you're running.")

    return parser


def codec_from_ext(path):
    import os
    import cv2
    basename, ext = os.path.splitext(path)
    assert ext in ['.mp4', '.avi', '.mkv'], "The provided video extension should be either 'mp4', 'avi' or 'mkv'. \
                                            Please check the 'output_video' argument."
    if ext == '.mp4': return cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.mkv': return cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
    elif ext == '.avi': return cv2.VideoWriter_fourcc(*"XVID")


def make_folder(path):
    import os
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_path, path)
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)  # also create parent directory(-ies) if needed


def recursive_clean(folder):
    import os
    import glob
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_path, folder)
    
    if os.path.isdir(folder) and os.path.exists(folder):
        
        folder = folder if str(folder).endswith('/') else folder + '/'
        file_list = glob.glob(folder + '*')
        
        for f in file_list:
            if os.path.isdir(f):
                recursive_clean(f)
            else:
                os.remove(f)
    
        os.rmdir(folder)
