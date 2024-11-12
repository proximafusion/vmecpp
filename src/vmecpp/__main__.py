from vmecpp._vmecpp import run, VmecINDATAPyWrapper
import sys

indata = VmecINDATAPyWrapper.from_file(sys.argv[1])
run(indata).wout
