import os

# Specify the path to the main folder containing all subfolders
main_folder_path = "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__lr_0.0001__conv__CLIP__use_reconstruction_True"

# Iterate over all folders inside the main folder
for subfolder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder_name)
    
    # Ensure we are only processing directories
    if os.path.isdir(subfolder_path):
        # Construct the full path to 'model.pth' inside the current subfolder
        model_path = os.path.join(subfolder_path, 'model.pth')
        
        # Check if the file exists and delete it
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Deleted: {model_path}")
        else:
            print(f"No model.pth file found in: {subfolder_path}")
