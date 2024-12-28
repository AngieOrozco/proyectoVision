import os
import cv2
import numpy as np
from typing import List


def load_images(filenames: List[str]) -> List[np.ndarray]:
    images = []
    for filename in filenames:
        img = cv2.imread(filename)
        if img is None:
            print(f"Warning: Unable to load image {filename}")
        else:
            images.append(img)
    return images

def to_gray(imgs: List[np.ndarray]) -> List[np.ndarray]:
    list_imgs_gray = []
    for idx, img in enumerate(imgs):
        try:
            imgs_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            list_imgs_gray.append(imgs_gray)
        except cv2.error as e:
            print(f"Error converting image {idx+1} to grayscale: {e}")
    return list_imgs_gray

def refine_corners(list_imgs_gray: List[np.ndarray], corners: List[np.ndarray], criteria: tuple) -> List[np.ndarray]:
    return [
        cv2.cornerSubPix(img, cor[1], (11, 11), (-1, -1), criteria) if cor[0] else [] 
        for img, cor in zip(list_imgs_gray, corners)
    ]

def draw_corners(images: List[np.ndarray], corners: List[np.ndarray], pattern_size: tuple) -> List[np.ndarray]:
    images_with_corners = []
    for img, corner in zip(images, corners):
        if len(corner) > 0:
            img_with_corners = cv2.drawChessboardCorners(img, pattern_size, corner, True)
            images_with_corners.append(img_with_corners)
    return images_with_corners


def show_image(window_name: str, image: np.ndarray):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_image(filename: str, image: np.ndarray):
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    success = cv2.imwrite(filename, image)
    if success:
        print(f"Image saved successfully: {filename}")
    else:
        print(f"Failed to save image: {filename}")

def get_chessboard_points(chessboard_shape: tuple, dx: float, dy: float) -> np.ndarray:
    coor = []
    n_cols, n_rows = chessboard_shape
    for i in range(n_rows):
        for j in range(n_cols):
            coor.append([j * dx, i * dy, 0])
    return np.array(coor, dtype=np.float32)

nums = [18, 27, 30, 34, 38, 50, 70, 81, 98, 99, 101, 106, 107, 108, 110, 113, 129, 134, 135, 149, 150, 152, 156, 158]
photos_path = "/home/pi/fotos/"
imgs_path = [os.path.join(photos_path, f"photo_{num}.jpg") for num in nums]

imgs = load_images(imgs_path)
if len(imgs) == 0:
    print("No images loaded. Check the file paths.")
    exit()
else:
    print(f"{len(imgs)} images loaded successfully.")


list_imgs_gray = to_gray(imgs)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_list = [cv2.findChessboardCorners(img, (7, 7)) for img in list_imgs_gray]

corners_refined = refine_corners(list_imgs_gray, corners_list, criteria)


imgs_with_corners = draw_corners(imgs, corners_refined, (7, 7))
for idx, img in enumerate(imgs_with_corners):
    output_path = f"output/corners_detected_{idx+1}.jpg"
    write_image(output_path, img)

chessboard_points = get_chessboard_points((7, 7), 30, 30)

valid_corners = [corner[1] for corner in corners_list if corner[0]]
valid_chessboard_points = [chessboard_points for _ in range(len(valid_corners))]


if valid_corners:
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        valid_chessboard_points, valid_corners, list_imgs_gray[0].shape[::-1], None, None
    )

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("RMS error:\n", rms)

    for idx, img in enumerate(imgs):
        undistorted_img = cv2.undistort(img, intrinsics, dist_coeffs, None, intrinsics)
        output_path = os.path.join("corrected_images", f"undistorted_image_{idx+1}.jpg")
        write_image(output_path, undistorted_img)
        show_image(f"Undistorted Image {idx+1}", undistorted_img)
else:
    print("No valid corners detected. Camera calibration skipped.")
