# Human-Activity-Recognition
# 📌 Project Overview
This project focuses on human activity recognition (HAR) from still images using machine learning and computer vision techniques. The dataset contains 15 activity classes, including calling, cycling, dancing, eating, running, and more, with 800 images per class. Our goal is to classify these activities using feature extraction methods and support vector machines (SVMs).

# 🔍 Exploratory Data Analysis (EDA)
Dataset: 15 activity classes with balanced distribution.
Image Sizes: Varied (7056p to 50625p), requiring resizing (64x64, 128x128, or 256x256).
Feature Importance: Assessed using random forest classification to determine impactful features.
Key Findings:
Color histograms showed some class separability.
Luminosity and contrast were not strong discriminators.
Certain classes (e.g., running vs. cycling) were harder to differentiate.

# 🛠 Feature Extraction & Filters
Various feature descriptors and filters were tested for improving classification:

Color Histogram ✅ (Partial class separation)
Sobel & Canny Edge Detection ✅ (Sobel performed better)
HOG (Histogram of Oriented Gradients) ✅ (Best single-feature performance)
LBP (Local Binary Patterns) ✅ (Texture-based detection)
SIFT (Scale-Invariant Feature Transform) ✅ (Keypoint-based features)
Patch Similarity ✅ (High correlation with activity patterns)

# 🏗 Machine Learning Model
Traditional Models: Multi-class SVM with RBF Kernel performed best.
Feature Engineering: PCA applied to reduce dimensionality while retaining 95% variance.
Best Feature Combination: HOG + LBP + SIFT + Color Histogram (35% accuracy).
# 🔥 CNN-Based Approach for Feature Extraction
Since handcrafted features had limited success, we utilized VGG16 (a deep learning model) for feature extraction.

Dimensionality Reduction: PCA to retain 95% variance.
Kernel SVM on Extracted Features:
Linear Kernel: 67.5% accuracy
RBF Kernel: 67.1% accuracy
# 🚀 Key Takeaways
Traditional filters and features were insufficient for high accuracy.
VGG16 feature extraction combined with SVM significantly improved performance.
Future Work: Experimenting with end-to-end deep learning models (CNNs) for better accuracy.
