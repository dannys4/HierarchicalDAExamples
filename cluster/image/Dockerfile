FROM debian

SHELL ["/bin/bash", "-c"]

# Dockerfile for dannys4/hierarchical_data_assimilation

ARG USERNAME=jovyan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

ENV JULIA_VERSION=1.11.5 \
    JULIA_INSTALLER=https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz \
    JULIA_BASE_DIR=/home/${USERNAME}

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    vim \
    python3 \
    && apt-get clean

USER $USERNAME

RUN wget --quiet ${JULIA_INSTALLER} -O /tmp/julia.tar.gz && \
    tar -xf /tmp/julia.tar.gz -C ${JULIA_BASE_DIR} && \
    rm /tmp/julia.tar.gz && \
    mv ${JULIA_BASE_DIR}/julia-${JULIA_VERSION} ${JULIA_BASE_DIR}/julia

ENV PATH=${JULIA_BASE_DIR}/julia/bin:$PATH

COPY --chown=${USER_UID}:${USER_GID} --chmod=777 ./Project.toml /home/${USERNAME}/.julia/environments/v1.11/
COPY --chown=${USER_UID}:${USER_GID} --chmod=777 ./examples/test.jl /home/${USERNAME}

RUN echo 'export PATH=${JULIA_BASE_DIR}/julia/bin:$PATH' >> /home/${USERNAME}/.bashrc && \
    echo 'export JULIA_CPU_TARGET="generic;znver3,clone_all;znver4,base(1)"' >> /home/${USERNAME}/.bashrc

RUN JULIA_CPU_TARGET="generic;znver3,clone_all;znver4,base(1)" julia -O3 -e "using Pkg; Pkg.instantiate()"
RUN JULIA_CPU_TARGET="generic;znver3,clone_all;znver4,base(1)" julia -O3 -e "using Pkg; Pkg.precompile()"
RUN JULIA_CPU_TARGET="generic;znver3,clone_all;znver4,base(1)" julia -O3 /home/${USERNAME}/test.jl 

WORKDIR /home/${USERNAME}

CMD ["/bin/bash"]
