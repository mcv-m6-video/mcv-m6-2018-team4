# Week 3 - Code Explanation
- **General:**
  - ../dataset.py: class that reads the specific frames of a given dataset.
  - ../evaluation.py: some useful functions for evaluate the resluts.

- **Task 1:** Optical Flow
  - **Task 1.1:** Block Matching
    - week4.py: Launcher of the task 1.1.
    - week4.py: Function for search for the best configuration (grid_search_block_matching(), plot_grid_search()).
    - block_matching.py: Function that computes the optical flow between two images.
    - flow_utils_w4.py: Useful functions for read, visualize and evaluate optical flow

  - **Task 1.2:** Block Matching vs Other Techniques
    - week4.py: Launcher of the task 1.2.
    - week4.py: Evaluation functions for task 2.1 and 2.2 (precision_recall_curve(), auc_vs_p(), f1score_alpha() and auc_all() functions).
    - cv2.calcOpticalFlowFarneback(): OpenCV functions that compute the optical flow using Farneback.

- **Task 2:** Video Stabilization
  - **Task 2.1:** Video Stabilization with BLock Matching
    - week4.py: Launcher of the task 2.1
    - week4.py: precision_recall_curve() function for evaluation
    - video_stabilization.py: Function for stabilize a video sequence

  - **Task 2.2:** BM Video Stabilization vs other techniques
    - week4.py: Launcher of the task 2.2
    - week4.py: precision_recall_curve() function for evaluation
    - week4.py: get_valid_mask() function for extractc the black borders given by the stabilization
    - video_stabilization/Chen Jia/: Video stabilization with Chen Jia method.
    - video_stabilization/ORB_descriptor/: Video stabilization with ORB descriptors method.

  - **Task 2.3:** Stabilize your own video
    - week4.py: Launcher of the task 2.2
    - video_stabilization.py: Function for stabilize a video sequence

