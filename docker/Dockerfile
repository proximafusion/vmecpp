# run as:
# docker build -t ghcr.io/proximafusion/vmecpp:latest  .
FROM ubuntu:22.04

RUN apt-get -q update && \
    apt-get -q -y install \
    build-essential \
    cmake \
    libnetcdf-dev \
    liblapack-dev \
    libomp-dev \
    libhdf5-dev \
    python3-pip \
    python-is-python3 \
    git \
    wget \
    ninja-build \
    gfortran

RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.24.1/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/local/bin/bazel \
    && chmod u+x /usr/local/bin/bazel

RUN git clone https://github.com/proximafusion/vmecpp.git
WORKDIR /vmecpp
RUN python -m pip install .
