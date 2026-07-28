import cv2
import numpy as np
import os
import pandas as pd


# =====================================
# Create Output Folder
# =====================================

output_folder = "output/21_shape_analysis"
os.makedirs(output_folder, exist_ok=True)


# =====================================
# Shape Feature Extraction Function
# =====================================

def extract_shape_features(image):

    if image is None:
        print("❌ Error: Image not found!")
        return None

    print("✅ Leaf image loaded successfully.")

    # -------------------------------
    # Convert to Grayscale
    # -------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Binary Threshold
    # -------------------------------

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # -------------------------------
    # Find Contours
    # -------------------------------

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("❌ No leaf detected.")
        return None

    # -------------------------------
    # Largest Contour
    # -------------------------------

    largest_contour = max(contours, key=cv2.contourArea)

    # -------------------------------
    # Shape Features
    # -------------------------------

    area = cv2.contourArea(largest_contour)

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    # -------------------------------
    # Draw Contour
    # -------------------------------

    result = image.copy()

    cv2.drawContours(
        result,
        [largest_contour],
        -1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        f"Area : {area:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"Perimeter : {perimeter:.2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # -------------------------------
    # Save Image
    # -------------------------------

    output_image = os.path.join(
        output_folder,
        "shape_analysis.jpg"
    )

    cv2.imwrite(output_image, result)

    # -------------------------------
    # Save CSV
    # -------------------------------

    df = pd.DataFrame({
        "Area": [area],
        "Perimeter": [perimeter]
    })

    csv_path = os.path.join(
        output_folder,
        "shape_features.csv"
    )

    df.to_csv(csv_path, index=False)

    # -------------------------------
    # Print Results
    # -------------------------------

    print("\n========== Shape Features ==========")

    print(f"Leaf Area      : {area:.2f} pixels²")
    print(f"Leaf Perimeter : {perimeter:.2f} pixels")

    print("\n✅ Shape Analysis Completed")

    print(f"📷 Image Saved : {output_image}")
    print(f"📄 CSV Saved   : {csv_path}")

    return {
        "Area": area,
        "Perimeter": perimeter
    }


# =====================================
# Main Function
# =====================================

if __name__ == "__main__":

    image_path = "../../dataset/test/hibiscus1.jpeg"

    image = cv2.imread(image_path)

    features = extract_shape_features(image)

    if features is not None:
        print("\nReturned Features:")
        print(features)