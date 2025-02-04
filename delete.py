import os
import shutil

COMP = 1
# Specify the path to the main folder containing all subfolders
main_folder_path = "plots/Flying"
for f in os.listdir(main_folder_path):
    folder_name = os.path.join(main_folder_path, f)
    # Iterate over all folders inside the main folder
    for subfolder_name in os.listdir(folder_name):
        subfolder_path = os.path.join(folder_name, subfolder_name)
        # Ensure we are only processing directories
        if os.path.isdir(subfolder_path) and (subfolder_name.endswith("bad") or "correct_F" not in subfolder_name):
            comp0 = os.path.join("/mnt/sda2/Alon", subfolder_path)
            comp1 = os.path.join("/mnt_hdd15tb/alonkay/Thesis/", subfolder_path)
            if COMP==0 and os.path.exists(comp0):
                # shutil.rmtree(comp0)
                print(f"Deleted: {comp0}")
            elif COMP==1 and os.path.exists(comp1):
                # shutil.rmtree(comp1)
                print(f"Deleted: {comp1}")
            elif COMP==2 and os.path.exists(subfolder_path):
                # shutil.rmtree(subfolder_path)
                print(f"Deleted: {subfolder_path}")


            # model_path = os.path.join(subfolder_path, 'model.pth')
            # backup_path = os.path.join(subfolder_path, 'backup_model.pth')
            # # Check if the file exists and delete it
            # if os.path.exists(model_path):
            #     # os.remove(model_path)
            #     print(f"Deleted: {model_path}")
                
            # if os.path.exists(backup_path):
            #     # os.remove(backup_path)
            #     print(f"Deleted: {backup_path}")
