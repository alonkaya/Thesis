FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# RUN apt-get update -y && apt-get install -y python3.9 sudo mesa-libGL
RUN apt-get update -y && apt-get install -y python3.9 sudo libgl1-mesa-glx


# RUN useradd -m <username>

# RUN chown -R <username>:<username> /home/<username>/

COPY . /home/alonkay/app/

# USER <username>

RUN cd /home/alonkay/app/ && pip3 install -r requirements.txt

WORKDIR /home/alonkay/app

CMD ["python3", "Main.py"]