import os
import shutil

# Specify the path to the main folder containing all subfolders
main_folder_path = "plots/Stereo/Winners/"
for f in os.listdir(main_folder_path):
    folder_name = os.path.join(main_folder_path, f)
    # Iterate over all folders inside the main folder
    for subfolder_name in os.listdir(folder_name):
        subfolder_path = os.path.join(folder_name, subfolder_name)
        # Ensure we are only processing directories
        if os.path.isdir(subfolder_path) and subfolder_name.endswith("bad"):
            print(subfolder_path)
            # shutil.rmtree(subfolder_path)

            # model_path = os.path.join(subfolder_path, 'model.pth')
            # backup_path = os.path.join(subfolder_path, 'backup_model.pth')
            # # Check if the file exists and delete it
            # if os.path.exists(model_path):
            #     # os.remove(model_path)
            #     print(f"Deleted: {model_path}")
                
            # if os.path.exists(backup_path):
            #     # os.remove(backup_path)
            #     print(f"Deleted: {backup_path}")
