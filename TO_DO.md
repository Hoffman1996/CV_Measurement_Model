# **To-Do:**
1. Create script for dataset preprocessing:
-   if label has more than 4 points
-   if label is set not as top-left going clockwise
-   remove duplicats in dataset (even when the have different names)?
-   perspective correction - do we need it? if so, to which images (user's input and or dataset images - test, train, val), how, when, and why?
-   remove images with more than 1 object

2. Training the model:
-   consider changing image size for training - 1280? more?
-   change num of epoches and batches?
-   check which augment and train func varients to keep/remove/modify
-   does optimizer affect lr?
-   what to look for in training results? how to read the graphs and what does the mAPs teach?

3. Predict on test set:
-   check which augment and predict func varients to keep/remove/modify
-   what to look for in predict results?




# to review:
2. Training Analysis
Image size considerations:
1280px might help but has trade-offs:

Pros: Better small detail detection, potentially more accurate measurements
Cons: 3x longer training time, requires batch size reduction (memory limits)
Recommendation: Test 1280px on a subset first. Your current 1024px with millimeter accuracy target might be sufficient.

Epochs/batches:
Your current setup (150 epochs, batch=8) is reasonable for 1400 images. Watch for:

Early stopping around epoch 100-120 suggests good convergence
If loss plateaus early, reduce learning rate or add regularization
If still improving at epoch 150, extend to 200 epochs

Augmentation review:
Your current settings are well-tuned for precision tasks:

scale=0.2 - good for geometric precision
degrees=5.0 - conservative rotation for windows/doors
mosaic=0.5, close_mosaic=10 - good balance
Keep as is - these are optimized for your millimeter accuracy requirement

Optimizer and learning rate:
AdamW doesn't directly affect learning rate values, but:

It affects how learning rate changes are applied
Your lr0=0.005, lrf=0.01 with cosine scheduling is appropriate
AdamW's weight decay helps prevent overfitting with small datasets

Reading training results:
Focus on these metrics:

mAP50: Most important for your use case (IoU=0.5 threshold)
Precision/Recall balance: High precision critical for measurement accuracy
Loss convergence: Should plateau, not oscillate
Validation vs training gap: Large gap indicates overfitting

3. Test Set Evaluation
Parameters to keep:

conf=0.8 - high threshold appropriate for measurement accuracy
iou=0.65 - good for preventing duplicates
max_det=1 - matches your single-window measurement use case

What to analyze:

Detection consistency: Same image should give same results
Edge cases: Windows at image boundaries, partial occlusion
Confidence distribution: Are detections consistently high confidence?
False negatives: More critical than false positives for measurement
Geometric accuracy: Do detected boxes align well with actual window edges?

Critical Missing Elements:
Ground truth validation: You mentioned manual validation but haven't implemented it. Create a script to:

Compare predictions against known window dimensions
Calculate measurement error distribution
Identify systematic biases in your measurements

Error analysis: Track where your system fails:

ChArUco detection failures
Window detection failures
Measurement outliers

Robustness testing: Test with:

Different lighting conditions
Various ChArUco board positions
Different window materials/colors

Your overall approach is sound, but focus on measurement accuracy validation rather than just detection metrics. The real test is whether your millimeter measurements are actually accurate in practice.