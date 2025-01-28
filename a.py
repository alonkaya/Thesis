import os

for line in open("models.txt"):
    line = line.strip()
    print(line)
    if os.path.exists(line):
        print(f'removing {line}')
        # os.remove(line)