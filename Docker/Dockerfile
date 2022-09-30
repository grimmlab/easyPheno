FROM tensorflow/tensorflow:2.8.2-gpu
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip  &&  \
    apt-get install -y r-base && apt-get install -y vim && apt-get install -y git
RUN mkdir /configfiles
COPY docker_requirements.txt /configfiles
RUN pip3 install -r /configfiles/docker_requirements.txt
RUN pip3 install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN Rscript -e "install.packages('BGLR')"
