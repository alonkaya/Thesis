import os

for line in open("models.txt"):
    line = line.strip()
    if os.path.exists(line):
        print(f'removing {line}')
        os.remove(line)
    else:
        print(f'does not exist {line}')