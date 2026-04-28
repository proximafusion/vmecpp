{
  description = "Nix development shell for vmecpp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { nixpkgs, ... }:
    let
      lib = nixpkgs.lib;
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python313;
          hdf5 = pkgs.hdf5-fortran;
          hdf5Merged = pkgs.symlinkJoin {
            name = "hdf5-merged";
            paths = [
              hdf5
              hdf5.dev
              hdf5.bin
            ];
            postBuild = ''
              rm "$out/lib/cmake/hdf5-config.cmake"
              sed \
                -e 's|"''${CMAKE_CURRENT_LIST_DIR}/\.\./\.\./\.\./[^"]*"|"'"$out"'"|' \
                -e 's|''${PACKAGE_PREFIX_DIR}//nix/store/[a-z0-9]*-hdf5-cpp-fortran-[^/]*/|''${PACKAGE_PREFIX_DIR}/|g' \
                -e 's|"//nix/store/[a-z0-9]*-hdf5-cpp-fortran-[^/]*/|"'"$out"'/|g' \
                "${hdf5.dev}/lib/cmake/hdf5-config.cmake" \
                > "$out/lib/cmake/hdf5-config.cmake"

              for f in "$out"/lib/cmake/hdf5-targets*.cmake; do
                rm "$f"
                sed \
                  -e 's|${hdf5}|'"$out"'|g' \
                  -e 's|${hdf5.dev}|'"$out"'|g' \
                  "${hdf5.dev}/lib/cmake/$(basename "$f")" \
                  > "$f"
              done
            '';
          };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python
              cmake
              ninja
              gcc
              gfortran
              pkg-config
              hdf5Merged
              netcdf
              netcdffortran
              blis
              lapack-reference
              fftw
              nodejs
              openmpi
              git
              git-lfs
            ];

            shellHook = ''
              export CC=${pkgs.gcc}/bin/gcc
              export CXX=${pkgs.gcc}/bin/g++
              export FC=${pkgs.gfortran}/bin/gfortran
              export CMAKE_GENERATOR=Ninja
              export HDF5_DIR=${hdf5Merged}/lib/cmake
              export CMAKE_ARGS="-DHDF5_DIR=${hdf5Merged}/lib/cmake -DCMAKE_DISABLE_FIND_PACKAGE_netCDF=TRUE -DBLAS_LIBRARIES=${pkgs.blis}/lib/libblas.so -DLAPACK_LIBRARIES=${pkgs.lapack-reference}/lib/liblapack.so"
              export LD_LIBRARY_PATH=${lib.makeLibraryPath [
                pkgs.stdenv.cc.cc
                pkgs.blis
                pkgs.lapack-reference
              ]}:''${LD_LIBRARY_PATH:-}
              vmecpp_pip_constraints="''${TMPDIR:-/tmp}/vmecpp-pip-constraints.txt"
              cat > "$vmecpp_pip_constraints" <<EOF
pydantic<2.13
EOF
              export PIP_CONSTRAINT="$vmecpp_pip_constraints"
              export PIP_BUILD_CONSTRAINT="$vmecpp_pip_constraints"
              export SKIP="pyright"
            '';
          };
        }
      );
    };
}
