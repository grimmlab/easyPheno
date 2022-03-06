FROM nvidia/cuda:11.1.1-base-ubuntu20.04
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip
RUN apt-get install -y vim
RUN apt-get install -y git
RUN mkdir /configfiles
COPY requirements.txt /configfiles
RUN pip3 install -r /configfiles/requirements.txt
RUN pip3 install torch==1.10.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html

