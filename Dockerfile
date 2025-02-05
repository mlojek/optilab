FROM ubuntu

WORKDIR /optilab
COPY . .

RUN apt update && apt install python3.12 python3-pip make --yes
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED
RUN make install_dependencies
