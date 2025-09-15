FROM ubuntu:22.04

# Fetch latest package lists.
RUN apt-get update

# Install git to be able to clone repo.
RUN apt-get install -y git

# Setup the LLVM sources in /llvm.
WORKDIR /llvm

# Make a shallow clone of the LLVM repo.
RUN git clone --depth 1 https://github.com/llvm/llvm-project.git

# Install more packages.
# Split up to allow caching of above step of cloning the llvm repo.
RUN apt-get install -y \
    python3 \
    gcc g++ \
    cmake \
    ninja-build

# Build clang and libomp with TSan/Archer support.
RUN cd llvm-project && \
    mkdir build && \
    cmake -G Ninja -S llvm -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt;libcxx;libcxxabi;libunwind;openmp" && \
    ninja -C build runtimes && \
    ninja -C build install \
    install-runtimes

# Install Bazel to be able to compile projects in the bind-mounted repo.
RUN apt-get install -y curl && curl -L "https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64" --output "/usr/bin/bazel" \
    && chmod a+x "/usr/bin/bazel"

# Install packages required for building VMEC++.
RUN apt-get install -y \
    libnetcdf-dev \
    liblapacke-dev \
    libfftw3-dev

# Install packages for in-docker development
RUN apt-get install -y vim

CMD [ "echo", "done" ]
