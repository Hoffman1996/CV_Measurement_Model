# **ROBOFLOW DATASET**
    In data.yaml file inside datesets folder:
        Make sure path to your train, test, and valid folders is the absolute path
        This evoids errors of unaccessable datasets folders during training and testing

# **ModuleNotFoundError**

####    Before trying anything else:
        Make sure that each of the packages in the root folder has an __init__.py file
        if not → create one for each of the packages and leave it empty
        this makes python know that this is a package and that its files can be imported elswhere

###  **Method 1 (Prefferable):**
###    Settings App (Windows 10/11)
        Settings → System → About
        Advanced system settings tab
        Environment Variables
        In System variables (bottom section):
            If PYTHONPATH exists: Select it → Edit
            If it doesn't exist: Click New
        Variable name: PYTHONPATH
        Variable value: C:\path\to\your\project (your project's root folder path)

###  **Method 2:**
###     Set PYTHONPATH to your project folder for the current PowerShell session only:
         $env:PYTHONPATH = "$PWD" (PWD id Path to Working Directory which is -> "C:\path\to\your\project")
         
###  **Method 3:**
###     Add project root to path programmatically
    At the top of your main scripts:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# **Fix device problem**
-     # if torch.cuda.is_available():
         # Make sure we expose GPU 0
         os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
         yolo_device = "0"  # explicit single-GPU
         print("✅ CUDA available. Using GPU 0.")
     else:
         yolo_device = "cpu"
         print("⚠️  CUDA not available to PyTorch at runtime. Falling back to CPU.")

