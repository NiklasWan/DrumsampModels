FROM tensorflow/tensorflow:2.5.0

RUN apt-get update

RUN apt-get install python3-pip ffmpeg libsndfile-dev -y

RUN useradd -ms /bin/bash dockerdev

USER dockerdev
WORKDIR /usr/local/bin/dockerdev

RUN pip3 install pylint 