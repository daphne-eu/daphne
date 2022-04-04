#!/bin/bash

# Copyright 2021 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stop immediately if any command fails.
set -e

#******************************************************************************
# Help message
#******************************************************************************

function printHelp {
    printLogo -t 0
    echo "Build the DAPHNE prototype."
    echo ""
    echo "Usage: $0 [-h|--help] [--target TARGET]"
    echo ""
    echo "This includes downloading and building all required third-party "
    echo "material, if necessary. Simply invoke this script without any "
    echo "arguments to build the prototype. You should only invoke it from "
    echo "the prototype's root directory (where this script resides)."
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help        Print this help message and exit."
    echo "  --target TARGET   Build the cmake target TARGET (defaults to '$target')"
    echo "  --clean           Remove all temporary build directories for a fresh build"
}

#******************************************************************************
# Build message helper
#******************************************************************************
daphne_red_fg='\e[38;2;247;1;70m'
daphne_red_bg='\e[48;2;247;1;70m'
daphne_blue_fg='\e[38;2;120;137;251m'
#daphne_blue_bg='\e[48;2;120;137;251m'
daphne_blue_bg='\e[48;2;36;42;76m'
daphne_black_fg='\e[38;2;0;0;0m'
daphne_black_bg='\e[48;2;0;0;0m'
reset='\e[00m'

daphne_msg() {
  local message date dotSize time timeFrame dots textSize
  ### time of animation
  timeFrame="0"
  if [ "$1" == "-t" ]; then
    timeFrame="$2"
    shift; shift
  fi
  prefix="[DAPHNE]"
  message="..${*}"
  date="[$(date +"%d.%m.%y %H:%M:%S")]"
  textSize=$(( $(tput cols) - ${#date}-${#prefix} ))
  dotSize=$(( ${textSize} - ${#message} ))

  dots=""
  for (( i = 0; i < dotSize; i++)); do
    dots="${dots}."
  done

  message="${message}${dots}"

  time=$( echo "scale=5; ${timeFrame}/${textSize}/1000" | bc )
  if [ "$timeFrame" -eq 0 ]; then
    printf "\r${daphne_red_fg}%*s\r${daphne_red_fg}%s${daphne_blue_fg}%s${daphne_red_fg}%s${reset}" "$(tput cols)" "${date}" "${prefix}" "${message}" "${date}"
  else
    printf "\r${daphne_red_fg}%*s\r${daphne_red_fg}%s${daphne_blue_fg}" "$(tput cols)" "${date}" "${prefix}"
    for ((i=0; i < textSize; i++)); do
      sleep "${time}"
      printf "%s" "${message:i:1}"
    done
    printf "${daphne_red_fg}%s${reset}" "${date}"
  fi
  printf "\n\r"
}

printLogo(){
  timeFrame="50"
  if [ "$1" == "-t" ]; then
    timeFrame="$2"
  fi
  daphne_msg -t "${timeFrame}" ""
  daphne_msg -t "${timeFrame}" ".Welcome to"
  daphne_msg -t "${timeFrame}" ""
  daphne_msg -t "${timeFrame}" ".._______.......................__"
  daphne_msg -t "${timeFrame}" ".|       \\.....................|  \\"
  daphne_msg -t "${timeFrame}" ".| \$\$\$\$\$\$\$\\  ______    ______  | \$\$____   _______    ______"
  daphne_msg -t "${timeFrame}" ".| \$\$  | \$\$ |      \\  /      \\ | \$\$    \\ |       \\  /      \\"
  daphne_msg -t "${timeFrame}" ".| \$\$  | \$\$  \\\$\$\$\$\$\$\\|  \$\$\$\$\$\$\\| \$\$\$\$\$\$\$\\| \$\$\$\$\$\$\$\\|  \$\$\$\$\$\$\\"
  daphne_msg -t "${timeFrame}" ".| \$\$  | \$\$ /      \$\$| \$\$  | \$\$| \$\$  | \$\$| \$\$  | \$\$| \$\$    \$\$"
  daphne_msg -t "${timeFrame}" ".| \$\$__/ \$\$|  \$\$\$\$\$\$\$| \$\$__/ \$\$| \$\$  | \$\$| \$\$  | \$\$| \$\$\$\$\$\$\$\$"
  daphne_msg -t "${timeFrame}" ".| \$\$    \$\$ \\\$\$    \$\$| \$\$    \$\$| \$\$  | \$\$| \$\$  | \$\$ \\$\$     \\"
  daphne_msg -t "${timeFrame}" ". \\\$\$\$\$\$\$\$   \\\$\$\$\$\$\$\$| \$\$\$\$\$\$\$  \\\$\$   \\\$\$ \\\$\$   \\\$\$  \\\$\$\$\$\$\$\$"
  daphne_msg -t "${timeFrame}" ".....................| \$\$"
  daphne_msg -t "${timeFrame}" ".....................| \$\$"
  daphne_msg -t "${timeFrame}" "......................\\\$\$..................EU-H2020.//.957407"
  daphne_msg -t "${timeFrame}" ""
  printf "\n\n"
}

#******************************************************************************
# Clean build directories
#******************************************************************************

function cleanBuildDirs {
    echo "-- Cleanup of build directories pwd=$(pwd) ..."

    cd "$projectRoot"

    local dirs=("build" \
        "thirdparty/llvm-project/build" \
        "thirdparty/antlr" \
        "thirdparty/catch2" \
        "thirdparty/OpenBLAS" \
        "thirdparty/grpc")
    local files=(\
        "thirdparty/antlr_v"*".install.success" \
        "thirdparty/catch2_v"*".install.success" \
        "thirdparty/grpc_v"*".install.success" \
        "thirdparty/openBlas_v"*".install.success" \
        "thirdparty/llvm_v"*".install.success" \
        "${llvmCommitFilePath}")

    echo -e "${daphne_red_fg}WARNING.${reset} This will delete following..."
    echo "Directories:"
    for dir in "${dirs[@]}"; do
        echo " > $dir"
    done
    echo "Files:"
    for file in "${files[@]}"; do
        echo " > $file"
    done

    echo

    read -p "Are you sure? (y/n) " answer

    if [[ "$answer" != [yY] ]]; then
        echo "Abort."
        exit 0
    fi


    # Delete entire directories.
    for dir in "${dirs[@]}"; do
        if [ -d "${dir}" ]
        then
            echo "---- cleanup ${dir}"
            rm -rf "${dir}"
        else
            echo "---- cleanup ${dir} - non-existing"
        fi
    done

    # Delete individual files.

    for file in "${files[@]}"; do
      if [ -f "$file" ]
      then
          echo "---- cleanup $file"
          rm -f "$file"
      else
          echo "---- cleanup $file - non-existing"
      fi
    done

}

#******************************************************************************
# Create Indicator-files
#******************************************************************************

#// creates indicator file which indicates successful dependency installation in <projectRoot>/thirdparty/
#// param 1 dependency name
dependency_install_success() {
  daphne_msg "Successfully installed ${1}."
  touch "${thirdpartyPath}/${1}.install.success"
}

#// checks if dependency is installed successfully
#// param 1 dependency name
is_dependency_installed() {
  [ -e "${thirdpartyPath}/${1}.install.success" ]
}

#******************************************************************************
# Set some paths
#******************************************************************************

projectRoot=$(pwd)
thirdpartyPath=$projectRoot/thirdparty
llvmCommitFilePath=$thirdpartyPath/llvm-last-built-commit.txt

# a hotfix, to solve issue #216 @todo investigate possible side effects
installLibDir=lib

#******************************************************************************
# Parse arguments
#******************************************************************************

# Defaults.
target="daphne"

while [[ $# -gt 0 ]]
do
    key=$1
    shift
    case $key in
        -h|--help)
            printHelp
            exit 0
            ;;
        --clean)
            cleanBuildDirs
            exit 0
            ;;
        --target)
            target=$1
            shift
            ;;
        *)
            printf "Unknown option: '%s'\n\n" "$key"
            printHelp
            exit 1
            ;;
    esac
done;


if [ ! -d "${projectRoot}/build" ]; then
  printLogo
fi


# Make sure that the submodule(s) have been updated since the last clone/pull.
# But only if this is a git repo.
if [ -d .git ]
then
    git submodule update --init --recursive
fi


#******************************************************************************
# Download third-party material if necessary
#******************************************************************************

#------------------------------------------------------------------------------
# Antlr4 (parser)
#------------------------------------------------------------------------------
antlrDirName=antlr
antlrVersion=4.9.2
antlrJarName=antlr-${antlrVersion}-complete.jar
antlrCppRuntimeDirName=antlr4-cpp-runtime-${antlrVersion}-source
antlrCppRuntimeZipName=$antlrCppRuntimeDirName.zip

# Download and build antlr4 C++ run-time if it does not exist yet.
if ! is_dependency_installed "antlr_v${antlrVersion}"; then
    daphne_msg -t 700 "Get Antlr version ${antlrVersion}"
    mkdir --parents "${thirdpartyPath}/${antlrDirName}"
    cd "${thirdpartyPath}/${antlrDirName}"
    # Download antlr4 jar if it does not exist yet.
    if [ ! -f $antlrJarName ]
    then
      daphne_msg "Download Antlr v${antlrVersion} java executable"
      wget https://www.antlr.org/download/$antlrJarName
    fi
    if [ ! -f "$antlrCppRuntimeZipName" ]; then
      daphne_msg "Download Antlr v${antlrVersion} Runtime"
      rm -rf "${antlrCppRuntimeDirName}"
      mkdir --parents "${thirdpartyPath}/${antlrCppRuntimeDirName}"
      wget https://www.antlr.org/download/$antlrCppRuntimeZipName
      unzip "$antlrCppRuntimeZipName" -d "$antlrCppRuntimeDirName"
    fi

    # Github disabled the unauthenticated git:// protocol, patch antlr4 to use https://
    # until we upgrade to antlr4-4.9.3+
    sed -i 's#git://github.com#https://github.com#' runtime/CMakeLists.txt

    cd "${thirdpartyPath}/${antlrCppRuntimeDirName}"
    rm -rf ./build
    mkdir -p build
    mkdir -p run
    cd build
    # if building of antlr fails, because its unable to clone utfcpp, it is probably due to using prohibited protocol by github
    # to solve this change the used url of github by following command:
    # $ git config --global url."https://github.com/".insteadOf git://github.com/

    daphne_msg "Build Antlr v${antlrVersion}"
    cmake .. -G Ninja  -DANTLR_JAR_LOCATION=../$antlrJarName -DANTLR4_INSTALL=ON -DCMAKE_INSTALL_PREFIX=../run/usr/local -DCMAKE_INSTALL_LIBDIR=$installLibDir
    cmake --build . --target install
    dependency_install_success "antlr_v${antlrVersion}"
else
    daphne_msg "No need to build Antlr4 again."
fi


#------------------------------------------------------------------------------
# catch2 (unit test framework)
#------------------------------------------------------------------------------
# Download catch2 release zip (if necessary), and unpack the single header file
# (if necessary).
catch2Name=catch2
catch2Version=2.13.8 # for upgrades, it suffices to simply change the version here
catch2ZipName=v$catch2Version.zip
catch2SingleHeaderName=catch.hpp
if ! is_dependency_installed "catch2_v${catch2Version}"; then
    daphne_msg -t 700 "Get catch2 version ${catch2Version}"
    mkdir --parents "${thirdpartyPath}/${catch2Name}"
    cd "${thirdpartyPath}/${catch2Name}"
    if [ ! -f $catch2ZipName ] || [ ! -f $catch2SingleHeaderName ]
    then
      daphne_msg "Download catch2 version ${catch2Version}"
      wget https://github.com/catchorg/Catch2/archive/refs/tags/$catch2ZipName
      unzip -p $catch2ZipName "Catch2-$catch2Version/single_include/catch2/catch.hpp" \
          > $catch2SingleHeaderName
    fi
    dependency_install_success "catch2_v${catch2Version}"
else
    daphne_msg "No need to download Catch2 again."
fi


#------------------------------------------------------------------------------
# OpenBLAS (basic linear algebra subprograms)
#------------------------------------------------------------------------------
openBlasDirName=OpenBLAS
openBlasVersion=0.3.19
openBlasZipName=OpenBLAS-$openBlasVersion.zip
openBlasInstDirName=installed
if ! is_dependency_installed "openBlas_v${openBlasVersion}"; then
    daphne_msg -t 700 "Get OpenBlas version ${catch2Version}"
    mkdir --parents "${thirdpartyPath}/${openBlasDirName}"
    cd "${thirdpartyPath}/${openBlasDirName}"
    wget "https://github.com/xianyi/OpenBLAS/releases/download/v${openBlasVersion}/${openBlasZipName}"
    unzip "$openBlasZipName"
    mkdir --parents "$openBlasInstDirName"
    cd "OpenBLAS-${openBlasVersion}"
    make -j
    make install PREFIX=../$openBlasInstDirName
    dependency_install_success "openBlas_v${openBlasVersion}"
else
    daphne_msg "No need to build OpenBlas again."
fi


#------------------------------------------------------------------------------
# gRPC
#------------------------------------------------------------------------------
grpcDirName=grpc
grpcVersion=1.38.0
grpcInstDir="${thirdpartyPath}/${grpcDirName}/installed"
if ! is_dependency_installed "grpc_v${grpcVersion}"; then
    daphne_msg -t 700 "Get grpc version ${grpcVersion}"
    cd "${thirdpartyPath}"
    # Download gRPC source code.
    if [ -d "${thirdpartyPath}/${grpcDirName}" ]; then
      rm -rf "${thirdpartyPath}/${grpcDirName:?}"
    fi
    git clone --recurse-submodules -b v${grpcVersion} https://github.com/grpc/grpc $grpcDirName
    mkdir --parents "${grpcInstDir}"

    cd "${thirdpartyPath}/${grpcDirName}"

    # Install gRPC and its dependencies.
    mkdir -p "cmake/build"
    cd "cmake/build"
    daphne_msg "Build grpc version ${grpcVersion}"
    cmake \
      -DCMAKE_INSTALL_PREFIX="${grpcInstDir}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      ../..
      #-DgRPC_ABSL_PROVIDER=package \
    make -j4 install
    cd "${thirdpartyPath}/${grpcDirName}"

    # Install abseil.
    mkdir -p "third_party/abseil-cpp/cmake/build"
    cd "third_party/abseil-cpp/cmake/build"
    cmake -DCMAKE_INSTALL_PREFIX="${grpcInstDir}" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
        ../..
    make -j
    make install
    dependency_install_success "grpc_v${grpcVersion}"
else
    daphne_msg "No need to build GRPC again."
fi



#------------------------------------------------------------------------------
# Build MLIR
#------------------------------------------------------------------------------
# We rarely need to build MLIR/LLVM, only during the first build of the
# prototype and after upgrades of the LLVM sub-module. To avoid unnecessary
# builds (which take several seconds even if there is nothing to do), we store
# the LLVM commit hash we built into a file, and only rebuild MLIR/LLVM if this
# file does not exist (first build of the prototype) or does not contain the
# expected hash (upgrade of the LLVM sub-module).

llvmName=llvm-project
llvmCommit="llvmCommit-local-none"
if [ -e .git ] # Note: .git in the submodule is not a directory.
then
    llvmCommit=$(git log -1 --format=%H)
fi

if ! is_dependency_installed "llvm_v${llvmCommit}" || [ "$(cat "${llvmCommitFilePath}")" != "$llvmCommit" ]; then
    daphne_msg -t 700 "Build llvm version ${llvmCommit}"
    cd "${thirdpartyPath}/${llvmName}"
    echo "Need to build MLIR/LLVM."
    mkdir --parents "build"
    cd "build"
    cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=OFF \
       -DLLVM_TARGETS_TO_BUILD="X86" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DLLVM_ENABLE_LLD=ON \
       -DLLVM_ENABLE_RTTI=ON
    cmake --build . --target check-mlir
    echo "$llvmCommit" > "$llvmCommitFilePath"
    dependency_install_success "llvm_v${llvmCommit}"
else
    daphne_msg "No need to build MLIR/LLVM again."
fi


# *****************************************************************************
# Build the DAPHNE prototype.
# *****************************************************************************

daphne_msg -t 700 "Build Daphne"
mkdir --parents "${projectRoot}/build"
cd "${projectRoot}/build"
cmake -G Ninja .. \
    -DMLIR_DIR="$thirdpartyPath/$llvmName/build/lib/cmake/mlir/" \
    -DLLVM_DIR="$thirdpartyPath/$llvmName/build/lib/cmake/llvm/" \
    -DANTLR4_RUNTIME_DIR="$thirdpartyPath/$antlrDirName/$antlrCppRuntimeDirName" \
    -DANTLR4_JAR_LOCATION="$thirdpartyPath/$antlrDirName/$antlrJarName" \
    -DOPENBLAS_INST_DIR="$thirdpartyPath/$openBlasDirName/$openBlasInstDirName" \
    -DCMAKE_PREFIX_PATH="$grpcInstDir"\
    -DCMAKE_INSTALL_LIBDIR="$installLibDir"

# optional cmake flags (to be added to the command above):
# -DUSE_CUDA=ON
# -DCMAKE_BUILD_TYPE=Debug

cmake --build . --target "$target"
daphne_msg -t 700 "Successfully build Daphne//${target}"

set +e
