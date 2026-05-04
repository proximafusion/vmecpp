FROM ubuntu:22.04

# Fetch latest package lists.
RUN apt-get update

# Install git to be able to clone repo.
RUN apt-get install -y git

# Setup the LLVM sources in /llvm.
WORKDIR /llvm

# Pin to llvmorg-18.1.8 so the installed clang is version 18, matching the
# hardcoded include path /usr/local/lib/clang/18/include in the Bazel toolchain.
RUN git clone --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git

# Install more packages.
# Split up to allow caching of above step of cloning the llvm repo.
RUN apt-get install -y \
    python3 \
    gcc g++ \
    cmake \
    ninja-build

# Build clang and libomp with TSan/Archer support.
# install-clang installs the compiler to /usr/local/bin/clang (required by the
# Bazel toolchain config). install-runtimes installs libarcher and libomp with
# TSan support to /usr/local/lib.
RUN cd llvm-project && \
    mkdir build && \
    cmake -G Ninja -S llvm -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt;libcxx;libcxxabi;libunwind;openmp" && \
    ninja -C build install-clang install-runtimes

# Install Bazel to be able to compile projects in the bind-mounted repo.
RUN apt-get install -y curl && curl -L "https://github.com/bazelbuild/bazelisk/releases/download/v1.24.1/bazelisk-linux-amd64" --output "/usr/bin/bazel" \
    && chmod a+x "/usr/bin/bazel"

# Install packages required for building VMEC++.
RUN apt-get install -y \
    libnetcdf-dev \
    liblapacke-dev \
    libfftw3-dev

# Install packages for in-docker development
RUN apt-get install -y vim

CMD [ "echo", "done" ]
