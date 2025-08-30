# **ROBOFLOW DATASET**
- 

# **ModuleNotFoundError**
- $env:PYTHONPATH = "C:\Users\yuval\OneDrive - Afeka College Of Engineering\Final Project\CV_Model_ChArUco"
- ## change this path to your project's path


# **Fix device problem**
-     # if torch.cuda.is_available():
         # Make sure we expose GPU 0
         os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
         yolo_device = "0"  # explicit single-GPU
         print("✅ CUDA available. Using GPU 0.")
     else:
         yolo_device = "cpu"
         print("⚠️  CUDA not available to PyTorch at runtime. Falling back to CPU.")