FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# RUN apt-get update -y && apt-get install -y python3.9 sudo mesa-libGL
RUN apt-get update -y && apt-get install -y python3.9 libgl1-mesa-glx libglib2.0-0

# RUN useradd -m <username>

# RUN chown -R <username>:<username> /home/<username>/

COPY poses/ utils.py task.py requirements.txt params.py Main.py FunMatrix.py  FMatrixRegressor.py Dockerfile docker.yaml deepF_nocors.py Dataset.py CustomDataset.py /home/alonkay/app/

# USER <username>

RUN cd /home/alonkay/app/ && pip3 install -r requirements.txt

WORKDIR /home/alonkay/app

# CMD ["python3", "Main.py"]

