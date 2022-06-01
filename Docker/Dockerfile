FROM tensorflow/tensorflow:2.8.2-gpu
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip
RUN apt-get install -y r-base
RUN apt-get install -y vim
RUN apt-get install -y git
RUN mkdir /configfiles
COPY requirements.txt /configfiles
RUN pip3 install -r /configfiles/requirements.txt
RUN pip3 install torch==1.10.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN Rscript -e "install.packages('BGLR')"
