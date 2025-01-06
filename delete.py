import os

# Specify the path to the main folder containing all subfolders
main_folder_path = "plots/Affine/BS_32__lr_0.0001__alpha_10__conv__original_rotated_angle_30__shift_32"

for f in os.listdir(main_folder_path):
    p = os.path.join(main_folder_path, f)
    # Iterate over all folders inside the main folder
    for subfolder_name in os.listdir(p):
        subfolder_path = os.path.join(p, subfolder_name)
        
        # Ensure we are only processing directories
        if os.path.isdir(subfolder_path):
            # Construct the full path to 'model.pth' inside the current subfolder
            model_path = os.path.join(subfolder_path, 'model.pth')
            backup_path = os.path.join(subfolder_path, 'backup_model.pth')
            # Check if the file exists and delete it
            if os.path.exists(model_path):
                # os.remove(model_path)
                print(f"Deleted: {model_path}")
                
            if os.path.exists(backup_path):
                # os.remove(backup_path)
                print(f"Deleted: {backup_path}")
                