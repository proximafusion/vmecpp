An Ubuntu 22.04 image with VMEC++ installed as a global Python package.

## Get the image

```console
docker pull ghcr.io/proximafusion/vmecpp:latest
```

## Use the image

Simple test: run SIMSOPT's ["QH fixed resolution" example](/examples/simsopt_qh_fixed_resolution.py) with VMEC++:

```shell
docker run -it --rm ghcr.io/proximafusion/vmecpp:latest
# now inside the docker container (by default we'll be inside the vmecpp repo sources):
python examples/simsopt_qh_fixed_resolution.py
```

To run VMEC++ on configurations you have on your host system, e.g. in a directory `data_dir`,
you could mount that directory onto the docker container and use VMEC++'s CLI API:

```shell
docker run -it --rm -v/absolute/path/to/data_dir:/data_dir ghcr.io/proximafusion/vmecpp:latest
# now inside the docker container we can run the VMEC++ CLI:
python -m vmecpp /data_dir/input.xyz
```


## For developers: manually pushing a new image

1. create a GitHub token like in [Get the image](#get-the-image), but check the `write:packages` and `repo` permissions
2. log into the GitHub container registry, e.g. with `echo YOUR_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin`
3. build the docker image
  - `docker build --tag=ghcr.io/proximafusion/vmecpp:latest .`
  - on systems with newer docker, you might need `docker buildx build` instead of just `docker build`
4. push the docker image
  - `docker push ghcr.io/proximafusion/vmecpp:latest`
