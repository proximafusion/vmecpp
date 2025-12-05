# Background Info on TSan/Archer setup

Here is some background info on TSan:

- https://clang.llvm.org/docs/ThreadSanitizer.html
- https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual

While TSan apparently works fine with regular threads, OpenMP support is not as
widespread. This has the effect that a lot of false positives, to the point
where the output is not useful anymore, are generated when using TSan to check
OpenMP-parallelized programs. Essentially, TSan is by default not aware of the
impact of the various OpenMP constructs, like `#pragma omp barrier` or
`#pragma omp critical`.

A solution to this is available, though.

When searching for `tsan openmp` in your favourite search engine, you will
probably find this suggestion: https://stackoverflow.com/a/55147406 The author
suggests to build `libomp` with `-DLIBOMP_TSAN_SUPPORT=1`. This is
**deprecated** as of now.

Apparently, this functionality was subsequently externalized into its own
project: `https://github.com/PRUNERS/archer`. (This is not to be confused with a
UK supercomputing service http://archer.ac.uk/ !) More info on Archer can be
found here:

- https://pruners.github.io/archer/
- https://www.vi-hps.org/cms/upload/material/tw30/Archer.pdf (slide 8 and onwards)

Turns out, this is **deprecated** as well and Archer was integrated into the
LLVM project in the mean time.

This seems to be the **state of things as of Oct 2023**. Indeed, Archer can be
found in the `llvm` source tree:
https://github.com/llvm/llvm-project/tree/main/openmp/tools/archer

According to
[the README.md in there](https://github.com/llvm/llvm-project/tree/main/openmp/tools/archer#build-archer-within-clangllvm),
Archer is built by default with the OpenMP runtime: > This distribution of
Archer is automatically built with the OpenMP runtime and automatically loaded
by the OpenMP runtime.

The current Ubuntu LTS (22.04) has `clang-14` available:
https://packages.ubuntu.com/jammy/clang

## Setting up a first test case

TL;DR: This is an example commonly presented as a case for data races. As will
be seen when running this example, the errornous output due to a data race is
not reliably reproduced. Thus, skip this section if you are looking for an
example that breaks reliably.

Let's try to setup a test case and detect thread synchronization errors! Here is
the code (from
https://github.com/llvm/llvm-project/tree/main/openmp/tools/archer#example),
that should be saved in a directory of your choice as `myprogram.c`:

```C
#include <stdio.h>

#define N 1000

int main (int argc, char **argv) {
    int a[N];

    for (int i = 0; i < N; ++i) {
        a[i] = i;
    }

#pragma omp parallel for
    for (int i = 0; i < N - 1; ++i) {
        a[i] = a[i + 1];
    }

    printf("result: %d (expected: %d)\n", a[N - 2], a[N - 1]);

    return 0;
}
```

First check the compiler version:

```bash
clang --version
```

returns:

```
Ubuntu clang version 14.0.0-1ubuntu1.1
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/bin
```

Now we compile the example program:

```bash
clang -fsanitize=thread -fopenmp -g myprogram.c -o myprogram
```

First, run the program single-threaded to get the expected reference output:

```bash
OMP_NUM_THREADS=1 ./myprogram
```

prints:

```
result: 999 (expected: 999)
```

Now we can run the program with two threads and observe the data race error:

```bash
OMP_NUM_THREADS=2 ./myprogram
```

prints:

```
LLVMSymbolizer: error reading file: No such file or directory
==================
WARNING: ThreadSanitizer: data race (pid=376631)
  Read of size 4 at 0x7ffe669bf740 by main thread:
    #0 .omp_outlined._debug__ /home/jons/code/llvm_for_tsan/test_1/myprogram.c:14:16 (myprogram+0xcc356) (BuildId: 0066345fd78e8103927c837c88f6e131187985fb)
    #1 .omp_outlined. /home/jons/code/llvm_for_tsan/test_1/myprogram.c:12:1 (myprogram+0xcc405) (BuildId: 0066345fd78e8103927c837c88f6e131187985fb)
    #2 __kmp_invoke_microtask <null> (libomp.so.5+0xd2622) (BuildId: c33253376586768cf0f9935bb8ff20f50f293b50)
    #3 __libc_start_call_main csu/../sysdeps/nptl/libc_start_call_main.h:58:16 (libc.so.6+0x29d8f) (BuildId: a43bfc8428df6623cd498c9c0caeb91aec9be4f9)

  Previous write of size 4 at 0x7ffe669bf740 by thread T1:
    #0 .omp_outlined._debug__ /home/jons/code/llvm_for_tsan/test_1/myprogram.c:14:14 (myprogram+0xcc37b) (BuildId: 0066345fd78e8103927c837c88f6e131187985fb)
    #1 .omp_outlined. /home/jons/code/llvm_for_tsan/test_1/myprogram.c:12:1 (myprogram+0xcc405) (BuildId: 0066345fd78e8103927c837c88f6e131187985fb)
    #2 __kmp_invoke_microtask <null> (libomp.so.5+0xd2622) (BuildId: c33253376586768cf0f9935bb8ff20f50f293b50)

  Location is stack of main thread.

  Location is global '??' at 0x7ffe669a2000 ([stack]+0x1d740)

  Thread T1 (tid=376633, running) created by main thread at:
    #0 pthread_create <null> (myprogram+0x4d3dd) (BuildId: 0066345fd78e8103927c837c88f6e131187985fb)
    #1 <null> <null> (libomp.so.5+0xabe97) (BuildId: c33253376586768cf0f9935bb8ff20f50f293b50)
    #2 __libc_start_call_main csu/../sysdeps/nptl/libc_start_call_main.h:58:16 (libc.so.6+0x29d8f) (BuildId: a43bfc8428df6623cd498c9c0caeb91aec9be4f9)

SUMMARY: ThreadSanitizer: data race /home/jons/code/llvm_for_tsan/test_1/myprogram.c:14:16 in .omp_outlined._debug__
==================
result: 999 (expected: 999)
ThreadSanitizer: reported 1 warnings
```

The interesting part comes now, when we fix that data race and re-examine the
program with TSan. First, introduce a `#pragma omp critical` before the access
to `a` (quick fix, but should be ok for this example):

```C
#include <stdio.h>

#define N 1000

int main (int argc, char **argv) {
    int a[N];

    for (int i = 0; i < N; ++i) {
        a[i] = i;
    }

#pragma omp parallel for
    for (int i = 0; i < N - 1; ++i) {
#pragma omp critical
        a[i] = a[i + 1];
    }

    printf("result: %d (expected: %d)\n", a[N - 2], a[N - 1]);

    return 0;
}
```

However, after re-compiling and re-testing, we still get the TSan warnings.

It would be expected that, because of the data race, the code missing the
`omp critical` would sometimes print an incorrect result (result != 999). Note
that I was not able to get an incorrect result with this example, though. This
indicates that this test is not sensitive enough to data races to ~reliably
trigger them when they are introduced in the code.

## Second test case

We therefore turn to a second test case, which is easier to break. The code is
from the following presentation:
https://doku.lrz.de/files/11497064/11497067/1/1684602039657/OpenMP+Workshop+Day+3.pdf
(starting on PDF page 38)

```C
#include <stdio.h>
#include <math.h>

double f(double x) {
	return 4.0 / (1.0 + x * x);
}

double CalcPi(int n) {
	const double fH = 1.0 / ((double) n);
	double fSum = 0.0;
	double fX;
	int i;

#pragma omp parallel for private(fX,i) reduction(+:fSum)
	for (i = 0; i < n; ++i) {
		fX = fH * ((double) i + 0.5);
		fSum += f(fX);
	}

	return fH * fSum;
}

int main(int argc, char** argv) {
	double pi_approximation = CalcPi(100);
	printf("pi approx = %.6f\n", pi_approximation);
	printf("pi  exact = %.6f\n", M_PI);
	return 0;
}
```

Compile it (for now without TSan to not clobber the console when things to
sideways):

```bash
clang -fopenmp -g -lm pi_example.c -o pi_example
```

and run it:

```bash
OMP_NUM_THREADS=1 ./pi_example
```

prints:

```
pi approx = 3.141601
pi  exact = 3.141593
```

So far, this program is correct wrt. OpenMP thread synchronization and we can
run it with as many threads as we like and still get the correct output. For
example, run it with 16 threads:

```bash
OMP_NUM_THREADS=16 ./pi_example
```

prints:

```
pi approx = 3.141601
pi  exact = 3.141593
```

However, we can artificially break this program by commenting out the reduction
in line 14:

```C
#pragma omp parallel for private(fX,i) // reduction(+:fSum)
```

Now, compile it again:

```bash
clang -fopenmp -g -lm pi_example.c -o pi_example
```

Running it with a single thread still works:

```bash
OMP_NUM_THREADS=1 ./pi_example
```

still prints:

```
pi approx = 3.141601
pi  exact = 3.141593
```

However, running it with 16 threads now goes (sometimes) sideways:

```bash
OMP_NUM_THREADS=16 ./pi_example
```

prints:

```
pi approx = 2.870014
pi  exact = 3.141593
```

Now let's try to have TSan detect this error.

First, we want TSan to confirm that the original program (with the correct
`reduction` phrase active) is correct. Hence, revert the commenting-out of
`reduction` in line 14 and re-compile the program with TSan enabled:

```bash
clang -fsanitize=thread -fopenmp -g pi_example.c -o pi_example
```

Now run it (two threads should be enough to trigger the data race checks):

```bash
OMP_NUM_THREADS=2 ./pi_example
```

However, contrary to the expection, TSan report data races, even though the
result is always computed correctly:

```
LLVMSymbolizer: error reading file: No such file or directory
==================
WARNING: ThreadSanitizer: data race (pid=379187)
  Read of size 8 at 0x7ffcd591d708 by main thread:
    #0 CalcPi /home/jons/code/llvm_for_tsan/test_2/pi_example.c:20:14 (pi_example+0xcc247) (BuildId: 1c337c5c622e7b561f2ad7aae45631edfa56072b)
    #1 main /home/jons/code/llvm_for_tsan/test_2/pi_example.c:24:28 (pi_example+0xcc846) (BuildId: 1c337c5c622e7b561f2ad7aae45631edfa56072b)

  Previous atomic write of size 8 at 0x7ffcd591d708 by thread T1:
    #0 .omp_outlined._debug__ /home/jons/code/llvm_for_tsan/test_2/pi_example.c:14:1 (pi_example+0xcc67f) (BuildId: 1c337c5c622e7b561f2ad7aae45631edfa56072b)
    #1 .omp_outlined. /home/jons/code/llvm_for_tsan/test_2/pi_example.c:14:1 (pi_example+0xcc7f5) (BuildId: 1c337c5c622e7b561f2ad7aae45631edfa56072b)
    #2 __kmp_invoke_microtask <null> (libomp.so.5+0xd2622) (BuildId: c33253376586768cf0f9935bb8ff20f50f293b50)

  Location is stack of main thread.

  Location is global '??' at 0x7ffcd58fe000 ([stack]+0x1f708)

  Thread T1 (tid=379189, running) created by main thread at:
    #0 pthread_create <null> (pi_example+0x4d3fd) (BuildId: 1c337c5c622e7b561f2ad7aae45631edfa56072b)
    #1 <null> <null> (libomp.so.5+0xabe97) (BuildId: c33253376586768cf0f9935bb8ff20f50f293b50)
    #2 main /home/jons/code/llvm_for_tsan/test_2/pi_example.c:24:28 (pi_example+0xcc846) (BuildId: 1c337c5c622e7b561f2ad7aae45631edfa56072b)

SUMMARY: ThreadSanitizer: data race /home/jons/code/llvm_for_tsan/test_2/pi_example.c:20:14 in CalcPi
==================
pi approx = 3.141601
pi  exact = 3.141593
ThreadSanitizer: reported 1 warnings
```

Further testing of this behaviour can be done by:

- running above command with more threads: apart from more console cluttering
  due to false positives, the numerical result stays spot-on correct
- re-introducing the bug by commenting out `reduction` etc in line 14 and then
  re-compile: the bug re-appears

**This tells me that the stock `clang-14` provided with Ubuntu 22.04-LTS does not come with working OpenMP-capable TSan support.**

The solution to this is thus considered to be to build `libomp` (and `clang` in
turn) with TSan support from source.

## Building libomp and clang with TSan support

When looking for instructions on how to build `clang` and `libomp` of the LLVM
project, you will sooner than later come across this page of build instructions:
https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm
Some further insights can be gotten from the compilation instructions for
`libcxx`: https://libcxx.llvm.org/BuildingLibcxx.html#id4

Note that the corresponding website for `openmp` is rather scarce:

- https://openmp.llvm.org/
- https://openmp.llvm.org/SupportAndFAQ.html

but at least one can find the meeting minutes on LLVM-internal `libomp`
development:
https://openmp.llvm.org/SupportAndFAQ.html#openmp-in-llvm-technical-call

One challenge is to understand whether to build the `openmp` _runtime_ or
_project_. This was a topic on the previous discussion forum on LLVM:
https://discourse.llvm.org/t/openmp-project-or-runtime/70886

The suggestion is to build the runtime. AFAIK, this is what is referred to as a
_bootstrapping build_ in the `libcxx` build instructions:
https://libcxx.llvm.org/BuildingLibcxx.html#bootstrapping-build (In fact, the
discourse thread above is where I came across the much-more-useful `libcxx`
build instructions.) The key idea there is to build `clang` from sources and use
that to build `libomp`. This sounds attractive, as it allows to have a fully
self-consistent toolchain.

In order to have a platform-independent and reproducible build setup, we build
`clang` and `libomp` in a Docker container. This container is based on Ubuntu
22.04-LTS for being able to serve as a template for a native setup on that OS.
(Go after
https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository for
installing the official Docker binaries on an Ubuntu 22.04-LTS host.) The docker
container is available in the `llvm_docker` subfolder. Build it using `build.sh`
and then run a `bash` inside it using `run.sh`.

In the container, install your favourite editor (e.g. `apt-get install -y vim`)
and create `pi_example.c` from the template above.

Then build it with TSan support:

```bash
clang -fsanitize=thread -fopenmp -g -lm pi_example.c -o pi_example
```

and run it (for a start with only one thread):

```
OMP_NUM_THREADS=1 ARCHER_OPTIONS='verbose=1' ./pi_example
```

We get the following output:

```
Archer detected OpenMP application with TSan, supplying OpenMP synchronization semantics
Warning: please export TSAN_OPTIONS='ignore_noninstrumented_modules=1' to avoid false positive reports from the OpenMP runtime!
pi approx = 3.141601
pi  exact = 3.141593
```

Yeah, Archer is suddenly active and working!

Now, adjust the command line as requested:

```bash
OMP_NUM_THREADS=1 ARCHER_OPTIONS='verbose=1' TSAN_OPTIONS='ignore_noninstrumented_modules=1' ./pi_example
```

and we gete:

```
Archer detected OpenMP application with TSan, supplying OpenMP synchronization semantics
pi approx = 3.141601
pi  exact = 3.141593
```

Most importantly, we can now run the program with more than one thread and not
drown in false positives:

```bash
OMP_NUM_THREADS=16 ARCHER_OPTIONS='verbose=1' TSAN_OPTIONS='ignore_noninstrumented_modules=1' ./pi_example
```

which prints:

```
Archer detected OpenMP application with TSan, supplying OpenMP synchronization semantics
pi approx = 3.141601
pi  exact = 3.141593
```

So far, so good. But can this setup detect the artificial bug introduced when
removing the `reduction` statement in line 14 of `pi_example.c`?

Let's comment that `reduction` statement out and try again:

```bash
OMP_NUM_THREADS=2 ARCHER_OPTIONS='verbose=1' TSAN_OPTIONS='ignore_noninstrumented_modules=1' ./pi_example
```

which prints:

```
Archer detected OpenMP application with TSan, supplying OpenMP synchronization semantics
/usr/local/bin/llvm-symbolizer: error: '[stack]': No such file or directory
==================
WARNING: ThreadSanitizer: data race (pid=240)
  Write of size 8 at 0x7ffc07f18cb8 by thread T1:
    #0 CalcPi.omp_outlined_debug__ /root/pi_example.c:17:8 (pi_example+0xe3fa5)
    #1 CalcPi.omp_outlined /root/pi_example.c:14:1 (pi_example+0xe4075)
    #2 __kmp_invoke_microtask <null> (libomp.so+0xbe832)
    #3 CalcPi /root/pi_example.c:14:1 (pi_example+0xe3d3b)

  Previous write of size 8 at 0x7ffc07f18cb8 by main thread:
    #0 CalcPi.omp_outlined_debug__ /root/pi_example.c:17:8 (pi_example+0xe3fa5)
    #1 CalcPi.omp_outlined /root/pi_example.c:14:1 (pi_example+0xe4075)
    #2 __kmp_invoke_microtask <null> (libomp.so+0xbe832)
    #3 CalcPi /root/pi_example.c:14:1 (pi_example+0xe3d3b)
    #4 <null> <null> (libc.so.6+0x29d8f) (BuildId: a43bfc8428df6623cd498c9c0caeb91aec9be4f9)

  Location is stack of main thread.

  Location is global '??' at 0x7ffc07efa000 ([stack]+0x1ecb8)

  Thread T1 (tid=242, running) created by main thread at:
    #0 pthread_create /llvm/llvm-project/compiler-rt/lib/tsan/rtl/tsan_interceptors_posix.cpp:1022:3 (pi_example+0x5ec8b)
    #1 __kmp_create_worker <null> (libomp.so+0x9e2f2)

SUMMARY: ThreadSanitizer: data race /root/pi_example.c:17:8 in CalcPi.omp_outlined_debug__
==================
pi approx = 3.141601
pi  exact = 3.141593
ThreadSanitizer: reported 1 warnings
```

That is something we can work with!

## Build Bazel projects using our new custom clang+libomp

The challenge here is to get a custom Bazel compiler toolchain working, which is
then also capable of compiling dependencies, such as Abseil and `googletest`.

This is what we work on next.

## Build the stage1 demo using a custom toolchain

from: https://bazel.build/tutorials/ccp-toolchain-config

This uses a system-provided `clang` installation (instead of the default
compiler Bazel uses).

```bash
cd cpp-tutorial_stage1
bazel build --config=clang_config //main:hello-world
bazel-bin/main/hello-world
```

## Build the abseil-hello demo using the custom toolchain

from: https://github.com/abseil/abseil-hello/tree/master

This uses a system-provided `clang` installation (instead of the default
compiler Bazel uses).

```bash
cd cpp-tutorial_stage1
bazel build --config=clang_config //abseil-hello:hello_main
bazel-bin/abseil-hello/hello_main "from Abseil"
```

Fixed by added `-lm` to standard linker flags. This was inspired by:
https://github.com/bazelbuild/bazel/issues/934#issuecomment-193474914

Even the unit test works if a recent version of `googletest` is used:

```bash
bazel test --config=clang_config //abseil-hello:hello_test
```

## Environment with a self-built clang

This is required to use `archer` to ThreadSanitize OpenMP-parallelized programs.

```bash
cd with_selfbuilt_clang/llvm_docker
./build.sh
./run.sh
```

This sets up a docker container based on Ubuntu-22.04-LTS, in which a custom
`clang` is built from source. The `run.sh` command then launches a shell within
that container.

In that `bash`, one can build both of the above example programs:

```bash
cd /code
bazel build --config=clang_config //main:hello-world
bazel-bin/main/hello-world
```

as well as:

```bash
cd /code
bazel build --config=clang_config //abseil-hello:hello_main
bazel-bin/abseil-hello/hello_main "from Abseil"
```

However, the unit testing for the second example pulls in `googletest`, which
currently fails to build due to some include path problems. One can reproduce
this as follows within the container:

```bash
cd /code
bazel test --config=clang_config //abseil-hello:hello_test
```

## How it finally worked

We got help from @silvergasp on GitHub:

> The issue was that when clang was generating the `*.d` files (aka Makefile
> dependency files) it was expanding out to the full absolute path, which didn't
> match the include paths that bazel was internally tracking. This meant that
> bazel thought you were trying to include files that weren't tracked by bazel.
> This is considered an error in Bazel (this is generally a good thing for
> reproducible builds). To fix this we needed to tell `clang` to just leave the
> relative include paths alone in the dependency files rather than expanding
> them to absolute paths. To do this you simply need to provide
> `-no-canonical-prefixes` on the command line for `clang`.

Going forward:

> the particular functionality that teaches TSan about OpenMP (what should be
> archer AFAIK) is indeed somehow missing from the publicly available binaries.

Interestingly, archer used to be enabled by default in the older `llvm` deb
binaries (I have the `llvm-{15,17}` apt repos on my system that are not showing
`libarcher.so`) e.g.

```bash
$ apt-file find libarcher.so
libomp-10-dev: /usr/lib/llvm-10/lib/libarcher.so
libomp-11-dev: /usr/lib/llvm-11/lib/libarcher.so
libomp-12-dev: /usr/lib/llvm-12/lib/libarcher.so
```

However it is enabled by default in the latest non-apt binary releases e.g;

```bash
$ wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.4/clang+llvm-17.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
$ tar -xvf clang+llvm-17.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
$ find clang+llvm-17.0.4-x86_64-linux-gnu-ubuntu-22.04/ -name libarcher*
clang+llvm-17.0.4-x86_64-linux-gnu-ubuntu-22.04/lib/libarcher_static.a
clang+llvm-17.0.4-x86_64-linux-gnu-ubuntu-22.04/lib/libarcher.so
```

So it might be possible to just overlay that somehow in the docker image or
download it as a `http_archive`` directly using bazel.

Two separate repositories have been used in debugging this:

- https://github.com/jons-pf/tsan_with_openmp
- https://github.com/jons-pf/bazel_custom_cc_toolchain

## Misc

Delete the Bazel cache:

```bash
bazel clean --expunge
```

The PDF files of the slides mentioned in this articles are mirrored locally
here:

https://drive.google.com/drive/folders/1NBNTr4jDQy951CoG-AYqpDfKKpNKNSzh?usp=sharing
