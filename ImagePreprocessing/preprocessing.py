import cv2
import numpy as np
import os
from skimage.morphology import skeletonize

# ==============================================
# CREATE OUTPUT DIRECTORY
# ==============================================

output_folder = "output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ==============================================
# READ INPUT IMAGE
# ==============================================

image_path = "dataset/healthy/leaf5.jpeg"

img = cv2.imread(image_path)

if img is None:
    print("Error: Unable to read image.")
    print("Check the image path.")
    exit()

print("Image Loaded Successfully")

# ==============================================
# STEP 1 : ORIGINAL IMAGE
# ==============================================

cv2.imwrite(
    "output/1_original.jpg",
    img
)

# ==============================================
# STEP 2 : IMAGE QUALITY ASSESSMENT
# ==============================================

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

laplacian_variance = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

quality = img.copy()

cv2.putText(
    quality,
    "Sharpness : {:.2f}".format(laplacian_variance),
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

cv2.imwrite(
    "output/2_quality_assessment.jpg",
    quality
)

# ==============================================
# STEP 3 : RESIZE IMAGE
# ==============================================

img = cv2.resize(
    img,
    (512,512)
)

cv2.imwrite(
    "output/3_resized.jpg",
    img
)

# ==============================================
# STEP 4 : LAB COLOR SPACE
# ==============================================

lab = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2LAB
)

# Save only for visualization

lab_display = cv2.cvtColor(
    lab,
    cv2.COLOR_LAB2BGR
)

cv2.imwrite(
    "output/4_lab.jpg",
    lab_display
)

# ==============================================
# STEP 5 : CLAHE
# ==============================================

l,a,b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

l2 = clahe.apply(l)

lab2 = cv2.merge((l2,a,b))

clahe_image = cv2.cvtColor(
    lab2,
    cv2.COLOR_LAB2BGR
)

cv2.imwrite(
    "output/5_clahe.jpg",
    clahe_image
)

# ==============================================
# STEP 6 : BILATERAL FILTER
# ==============================================

bilateral = cv2.bilateralFilter(
    clahe_image,
    9,
    75,
    75
)

cv2.imwrite(
    "output/6_bilateral.jpg",
    bilateral
)

# ==============================================
# STEP 7 : AUTOMATIC ROI DETECTION
# ==============================================

gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = np.ones((5,5), np.uint8)

thresh = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest)

padding = 25

x = max(0, x-padding)
y = max(0, y-padding)
w = min(bilateral.shape[1]-x, w+padding*2)
h = min(bilateral.shape[0]-y, h+padding*2)

roi = bilateral[y:y+h, x:x+w]

cv2.imwrite(
    "output/7_roi.jpg",
    roi
)

# ==============================================
# STEP 8 : GRABCUT SEGMENTATION
# ==============================================

mask = np.zeros(roi.shape[:2], np.uint8)

bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)

rect = (
    10,
    10,
    roi.shape[1]-20,
    roi.shape[0]-20
)

cv2.grabCut(
    roi,
    mask,
    rect,
    bgModel,
    fgModel,
    5,
    cv2.GC_INIT_WITH_RECT
)

mask2 = np.where(
    (mask==2) | (mask==0),
    0,
    1
).astype("uint8")

grabcut = roi * mask2[:,:,np.newaxis]

cv2.imwrite(
    "output/8_grabcut.jpg",
    grabcut
)

# ==============================================
# STEP 9 : KEEP ONLY LARGEST LEAF
# ==============================================

gray_leaf = cv2.cvtColor(
    grabcut,
    cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray_leaf,
    10,
    255,
    cv2.THRESH_BINARY
)

contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest = max(contours, key=cv2.contourArea)

leaf_mask = np.zeros(binary.shape, dtype=np.uint8)

cv2.drawContours(
    leaf_mask,
    [largest],
    -1,
    255,
    thickness=cv2.FILLED
)

largest_leaf = cv2.bitwise_and(
    roi,
    roi,
    mask=leaf_mask
)

cv2.imwrite(
    "output/9_largest_leaf.jpg",
    largest_leaf
)

# ==============================================
# STEP 10 : HSV REFINEMENT
# ==============================================

hsv = cv2.cvtColor(
    largest_leaf,
    cv2.COLOR_BGR2HSV
)

lower_green = np.array([25,30,30])

upper_green = np.array([95,255,255])

green_mask = cv2.inRange(
    hsv,
    lower_green,
    upper_green
)

kernel = np.ones((3,3), np.uint8)

green_mask = cv2.morphologyEx(
    green_mask,
    cv2.MORPH_OPEN,
    kernel
)

green_mask = cv2.morphologyEx(
    green_mask,
    cv2.MORPH_CLOSE,
    kernel
)

refined_leaf = cv2.bitwise_and(
    largest_leaf,
    largest_leaf,
    mask=green_mask
)

cv2.imwrite(
    "output/10_refined_leaf.jpg",
    refined_leaf
)

# ==============================================
# STEP 11 : BINARY MASK
# ==============================================

gray_leaf = cv2.cvtColor(
    refined_leaf,
    cv2.COLOR_BGR2GRAY
)

_, binary_mask = cv2.threshold(
    gray_leaf,
    10,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    "output/11_binary_mask.jpg",
    binary_mask
)

# ==============================================
# STEP 12 : ADAPTIVE THRESHOLDING
# ==============================================

adaptive = cv2.adaptiveThreshold(
    gray_leaf,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    25,
    5
)

cv2.imwrite(
    "output/12_adaptive_threshold.jpg",
    adaptive
)

# ==============================================
# STEP 13 : TOP-HAT FILTERING
# ==============================================

kernel_tophat = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (15,15)
)

tophat = cv2.morphologyEx(
    gray_leaf,
    cv2.MORPH_TOPHAT,
    kernel_tophat
)

cv2.normalize(
    tophat,
    tophat,
    0,
    255,
    cv2.NORM_MINMAX
)

cv2.imwrite(
    "output/13_tophat.jpg",
    tophat
)

# ==============================================
# STEP 14 : MORPHOLOGICAL OPENING
# ==============================================

kernel_open = np.ones((3,3), np.uint8)

opening = cv2.morphologyEx(
    adaptive,
    cv2.MORPH_OPEN,
    kernel_open,
    iterations=1
)

cv2.imwrite(
    "output/14_opening.jpg",
    opening
)

# ==============================================
# STEP 15 : MORPHOLOGICAL CLOSING
# ==============================================

kernel_close = np.ones((5,5), np.uint8)

closing = cv2.morphologyEx(
    opening,
    cv2.MORPH_CLOSE,
    kernel_close,
    iterations=2
)

cv2.imwrite(
    "output/15_closing.jpg",
    closing
)

# ==============================================
# STEP 16 : VEIN ENHANCEMENT
# ==============================================

blur_large = cv2.GaussianBlur(
    gray_leaf,
    (31,31),
    0
)

vein_enhancement = cv2.subtract(
    gray_leaf,
    blur_large
)

vein_enhancement = cv2.normalize(
    vein_enhancement,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

cv2.imwrite(
    "output/16_vein_enhancement.jpg",
    vein_enhancement
)

# ==============================================
# STEP 17 : Skeletonization
# ==============================================

binary = vein_enhancement > 25

skeleton = skeletonize(binary)

skeleton = (skeleton.astype(np.uint8))*255

cv2.imwrite(
    "output/17_skeleton.jpg",
    skeleton
)

# ==============================================
# STEP 18 : Vein Overlay
# ==============================================

overlay = refined_leaf.copy()

overlay[skeleton==255] = [0,0,255]

cv2.imwrite(
    "output/18_vein_overlay.jpg",
    overlay
)

# ==============================================
# STEP 19 : Leaf Contour
# ==============================================

contours,_ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour_img = refined_leaf.copy()

cv2.drawContours(
    contour_img,
    contours,
    -1,
    (0,255,0),
    2
)

cv2.imwrite(
    "output/19_leaf_contour.jpg",
    contour_img
)

# ==============================================
# STEP 20 : Final Preprocessed Image
# ==============================================

cv2.imwrite(
    "output/20_final_leaf.jpg",
    refined_leaf
)