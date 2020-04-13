FROM python:3.7

MAINTAINER Benedikt Wiberg benedikt@luminovo.ai

RUN apt-get update
RUN apt-get install -y sudo

# create new user for service that is not root
ENV USER=hippo7
RUN useradd -Um ${USER}
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers;
USER $USER

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | sudo python
#ENV PATH="/home/${USER}/.poetry/bin:$PATH"
ENV PATH="/root/.poetry/bin:$PATH"
# do not create a virtualenv - install to system-wide python
RUN sudo poetry config virtualenvs.create false

WORKDIR /home/${USER}

# making use of docker cache
COPY ./pyproject.toml /home/${USER}
COPY ./poetry.lock /home/${USER}

RUN sudo apt-get update
RUN sudo apt-get install -y ffmpeg libglu1-mesa-dev mesa-common-dev freeglut3-dev > /dev/null
RUN sudo poetry install

COPY . /home/${USER}

RUN sudo poetry install

ENTRYPOINT [ "/bin/bash", "-c" ]
