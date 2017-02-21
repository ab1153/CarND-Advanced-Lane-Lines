from preprocess import *
from lane_finding import *

def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    undist_img = undistort(img, dist, mtx)

    hls = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS)
    saturation = hls[...,2]
    gray = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)
    gradx = sobel_abs(gray, (20,255), 5)
    grady = sobel_abs(gray, (10,255), 5, (0,1))
    sat_bin = thresh(saturation, (180,255))

    combined = np.zeros_like(sat_bin)
    cond = ( (sat_bin == 1) | ((gradx == 1) & (grady==1))  )
    combined[cond] = 1   

    combined_warp = cv2.warpPerspective(combined, pers_transform, combined.shape[::-1], flags=cv2.INTER_LINEAR)
    try:
        left, right, left_m, right_m = slide_window(combined_warp)
    except:
        return undist_img

    y_bottom = 720
    width = 1200
    left_radius, right_radius = calculate_radii(y_bottom, left_m, right_m)
    radius_curv = 0.5 * (left_radius + right_radius)    

    deviation = calculate_deviation(y_bottom, width, left_m, right_m)

    result = draw_info_unwrap(undist_img, pers_transform_inv, left, right, radius_curv, deviation)

    return result
    