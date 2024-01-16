FROM nvidia/cuda:12.3.1-base-rockylinux8

RUN yum update -y && yum install -y python3.9 python3-pip23.3.1 sudo \

RUN pip3 install --upgrade pip==23.3.1

# RUN useradd -m <username>

# RUN chown -R <username>:<username> /home/<username>/

COPY . /home/alonkay/app/

# USER <username>

RUN cd /home/alonkay/app/ && pip3 install -r requirements.txt

WORKDIR /home/alonkay/app