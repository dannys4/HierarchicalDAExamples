FROM debian

SHELL ["/bin/bash", "-c"]

ENV JULIA_VERSION=1.11.5 \
    JULIA_INSTALLER=https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz \
    JULIA_BASE_DIR=/opt

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    vim \
    python3 \
    && apt-get clean

RUN wget --quiet ${JULIA_INSTALLER} -O /tmp/julia.tar.gz && \
    tar -xf /tmp/julia.tar.gz -C ${JULIA_BASE_DIR} && \
    rm /tmp/julia.tar.gz && \
    mv ${JULIA_BASE_DIR}/julia-${JULIA_VERSION} ${JULIA_BASE_DIR}/julia

ENV PATH=${JULIA_BASE_DIR}/julia/bin:$PATH

COPY ./Project.toml /root/.julia/environments/v1.11/

RUN apt-get install -y tmux && \
    echo 'export PATH=/opt/julia/bin:$PATH' >> /root/.bashrc

RUN julia -e "using Pkg; Pkg.instantiate();"

WORKDIR /root

CMD ["/bin/bash"]