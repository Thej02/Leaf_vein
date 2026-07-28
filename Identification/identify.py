from ImagePreprocessing.preprocessing import preprocess_image

from shape_analysis import extract_shape_features

leaf = preprocess_image(image_path)

shape = extract_shape_features(leaf)