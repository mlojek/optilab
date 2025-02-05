FROM python:3.13

WORKDIR /optilab

COPY . .

RUN make install_dependencies

ENTRYPOINT /bin/bash
