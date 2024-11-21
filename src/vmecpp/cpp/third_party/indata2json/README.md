# Fully static binary for indata2json

Converts `input.CONFIG` Fortran files into the corresponding VMEC++-compatible `CONFIG.json`.

Original sources and usage instructions at https://github.com/jonathanschilling/indata2json.git.
Static binary built from master@4274976d0f1a5a978e9216b1938b3502b4250e75 on Ubuntu 22.04.

## Why did we commit a binary in the repo?

- the source code is Fortran, so very hard to build with our usual tools
- it's "only" 1.5M, less than several other files in the repo
- it's a self-contained statically linked binary, so it works on any reasonably recent Linux kernel, it does not require a specific OS
- it's rarely or never going to change or be updated: there won't be any need to ever diff this
- all alternatives seemed to have worse tradeoffs
