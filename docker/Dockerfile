# run as:
# env GITHUB_TOKEN=XYZ docker build --secret id=GITHUB_TOKEN .
# where XYZ is a github token that has permissions to read jons-pf/vmecpp.
FROM ubuntu:22.04

RUN apt-get -q update && \
    apt-get -q -y install build-essential libnetcdf-dev liblapacke-dev libopenmpi-dev python3-pip python-is-python3 git wget
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.24.1/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/local/bin/bazel \
    && chmod u+x /usr/local/bin/bazel

RUN --mount=type=secret,id=GITHUB_TOKEN,env=GITHUB_TOKEN git clone https://$GITHUB_TOKEN@github.com/jons-pf/vmecpp.git
WORKDIR /vmecpp
RUN python -m pip install .
