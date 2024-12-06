1. create a github token
  - go to https://github.com/settings/tokens
  - generate new token -> generate new token (classic)
  - tick repo and write:packages
  - click "generate token"
  - copy the token (this is your one chance)
2. log into the github container registry with that token, e.g.:
  - echo ghp_xyz | docker login ghcr.io -u jons-pf --password-stdin
3. build the docker image
  - env GITHUB_TOKEN=ghp_xyz docker build --tag=ghcr.io/jons-pf/vmecpp:latest --secret id=GITHUB_TOKEN .
4. push the docker image
  - docker push ghcr.io/jons-pf/vmecpp:latest
