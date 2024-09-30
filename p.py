# Open the file in read mode
import os


with open('a.txt', 'r') as file:
    # Iterate over each line in the file
    for line in file:
        par = os.path.dirname(line)
        dest = os.path.join(par, 'SED_0.5__L2_1__huber_1__lr_0.0001__conv__CLIP__use_reconstruction_True_drafts')
        print(f'from {line} to {dest}')
        # os.move(line, dest)