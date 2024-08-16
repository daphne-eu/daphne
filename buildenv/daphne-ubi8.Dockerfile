FROM redhat/ubi8

RUN dnf update -y --refresh

# dev tools
RUN dnf install -y redhat-lsb-core less nano git wget unzip file rsync vim xz

# build deps for DAPHNE deps
RUN dnf install -y gcc gcc-c++ pkg-config patch zstd libuuid-devel zlib-devel bzip2-devel

# build deps from epel
RUN dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
RUN dnf install -y lbzip2 ccache openssl-devel 
