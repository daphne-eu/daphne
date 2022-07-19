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

build_ts_begin=$(date +%s%N)

#******************************************************************************
# Help message
#******************************************************************************

function printHelp {
    printLogo
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
    echo "  --cleanAll        Remove all thirdparty library directories for a build from scratch"
    echo "  -nf, --no-fancy   Suppress all colored and animated output"
    echo "  -y, --yes         Accept all prompts"
    echo "  --arrow           Compile with support for Arrow/Parquet files"
}

#******************************************************************************
# Build message helper
#******************************************************************************
# Daphne Colors
daphne_red_fg='\e[38;2;247;1;70m'
daphne_red_bg='\e[48;2;247;1;70m'
daphne_blue_fg='\e[38;2;120;137;251m'
daphne_blue_bg='\e[48;2;36;42;76m'
daphne_black_fg='\e[38;2;0;0;0m'
daphne_black_bg='\e[48;2;0;0;0m'
reset='\e[00m'
fancy="1"

# Prints info message with style ... Daphne style 8-)
function daphne_msg() {
    local message date dotSize dots textSize columnWidth
    # get width of terminal
    columnWidth="$(tput cols)"
    # check if output is directed to file, if true, set width to 120
    if ! [ -t 1 ]; then
        columnWidth=120
    fi
    prefix="[DAPHNE]"
    message="..${*}"
    date="[$(date +"%d.%m.%y %H:%M:%S")]"
    textSize=$(( ${columnWidth} - ${#date}-${#prefix} ))
    dotSize=$(( ${textSize} - ${#message} ))

    dots=""
    for (( i = 0; i < dotSize; i++)); do
        dots="${dots}."
    done

    message="${message}${dots}"

    # no fancy output (if disabled or not standard output, e.g. piped into file)
    if [ "$fancy" -eq 0 ] || ! [ -t 1 ] ; then
        printf "%s%s%s\n" "${prefix}" "${message}" "${date}"
        return 0
    fi

    # colored output
    printf "${daphne_red_fg}%s${daphne_blue_fg}%s${daphne_red_fg}%s${reset}\n" "${prefix}" "${message}" "${date}"
    return 0
}

function printableTimestamp () {
    local t="0"
    local result=""
    local tmp
    if [ -n "$1" ]; then
        t="$1"
    fi

    if [ "$t" -eq 0 ]; then
        printf "0ns"
        return 0
    fi

    tmp=$((t % 1000))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}ns"
    fi
    # t < 1000 ns
    if [ "$t" -lt 1000 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 1000))" #us
    tmp=$((t % 1000))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}us${result}"
    fi
    # t < 1000 us
    if [ "$t" -lt 1000 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 1000))" #ms
    tmp=$((t % 1000))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}ms${result}"
    fi
    # t < 1000 ms
    if [ "$t" -lt 1000 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 1000))" #s
    tmp=$((t % 60))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}s${result}"
    fi
    # t < 60 s
    if [ "$t" -lt 60 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 60))" #min
    tmp=$((t % 60))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}min${result}"
    fi
    # t < 60 min
    if [ "$t" -lt 60 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 60))" #h
    result="${t}h${result}"
    printf "%s\n" "${result}"
}

function printLogo(){
    daphne_msg ""
    daphne_msg ".Welcome to"
    daphne_msg ""
    daphne_msg ".._______.......................__"
    daphne_msg ".|       \\.....................|  \\"
    daphne_msg ".| €€€€€€€\\  ______    ______  | €€____   _______    ______"
    daphne_msg ".| €€  | €€ |      \\  /      \\ | €€    \\ |       \\  /      \\"
    daphne_msg ".| €€  | €€  \\€€€€€€\\|  €€€€€€\\| €€€€€€€\\| €€€€€€€\\|  €€€€€€\\"
    daphne_msg ".| €€  | €€ /      €€| €€  | €€| €€  | €€| €€  | €€| €€    €€"
    daphne_msg ".| €€__/ €€|  €€€€€€€| €€__/ €€| €€  | €€| €€  | €€| €€€€€€€€"
    daphne_msg ".| €€    €€ \\€€    €€| €€    €€| €€  | €€| €€  | €€ \€€     \\"
    daphne_msg ". \\€€€€€€€   \\€€€€€€€| €€€€€€€  \\€€   \\€€ \\€€   \\€€  \\€€€€€€€"
    daphne_msg ".....................| €€"
    daphne_msg ".....................| €€"
    daphne_msg "......................\\€€..................EU-H2020.//.957407"
    daphne_msg ""
    printf "\n\n"
}

#******************************************************************************
# Clean build directories
#******************************************************************************

function clean() {
    # Throw error, if clean is executed, but output is piped to a file. In this case the user has to accept the
    # cleaning via parameter --yes
    if ! [ -t 1 ] && ! [ "$par_acceptAll" -eq 1 ]; then
        >&2 printf "${daphne_red_fg}"
        printf "Error: To clean Daphne while piping the output into a file set the --yes option, to accept cleaning.\n" | tee /dev/stderr
        >&2 printf "${reset}"
        exit 1
    fi

    cd "$projectRoot"
    local -n __dirs
    local -n __files

    if [ "$#" -gt 0 ]; then
        if [ -n "$1" ]; then
            __dirs=$1
        fi
        shift
    fi

    if [ "$#" -gt 0 ]; then
        if [ -n "$1" ]; then
            __files=$1
        fi
        shift
    fi
    if [ "$fancy" -eq 0 ] || ! [ -t 1 ] ; then
        printf "WARNING. This will delete the following..."
    else
        printf "${daphne_red_fg}WARNING.${reset} This will delete following..."
    fi
    echo "Directories:"
    for dir in "${__dirs[@]}"; do
        echo " > $dir"
    done
    echo "Files:"
    for file in "${__files[@]}"; do
        echo " > $file"
    done

    echo

    # prompt confirmation, if not set by --yes
    if [ "$par_acceptAll" == "0" ]; then
        read -p "Are you sure? (y/n) " answer

        if [[ "$answer" != [yY] ]]; then
            echo "Abort."
            exit 0
        fi
    fi

    # Delete entire directories.
    for dir in "${__dirs[@]}"; do
        if [ -d "${dir}" ]; then
            echo "---- cleanup ${dir}"
            rm -rf "${dir}"
        else
            echo "---- cleanup ${dir} - non-existing"
        fi
    done

    # Delete individual files.
    for file in "${__files[@]}"; do
        if [ -f "$file" ]; then
            echo "---- cleanup $file"
            rm -f "$file"
        else
            echo "---- cleanup $file - non-existing"
        fi
    done
}

# Cleans all build directories
function cleanBuildDirs() {
    echo "-- Cleanup of build directories in ${projectRoot} ..."

    local dirs=("${daphneBuildDir}" "${buildPrefix}" "${installPrefix}")
    local files=(\
      "${thirdpartyPath}/absl_v"*".install.success" \
      "${thirdpartyPath}/antlr_v"*".install.success" \
      "${thirdpartyPath}/catch2_v"*".install.success" \
      "${thirdpartyPath}/grpc_v"*".install.success" \
      "${thirdpartyPath}/nlohmannjson_v"*".install.success" \
      "${thirdpartyPath}/openBlas_v"*".install.success" \
      "${thirdpartyPath}/llvm_v"*".install.success" \
      "${thirdpartyPath}/arrow_v"*".install.success" \
      "${llvmCommitFilePath}")
    
    clean dirs files
}

# Cleans build directory and all dependencies
function cleanAll() {
    echo "-- Cleanup of build and library directories in ${projectRoot} ..."

    local dirs=("${daphneBuildDir}" "${buildPrefix}" "${sourcePrefix}" "${installPrefix}" "${cacheDir}")
    local files=(\
      "${thirdpartyPath}/absl_v"*".install.success" \
      "${thirdpartyPath}/absl_v"*".download.success" \
      "${thirdpartyPath}/antlr_v"*".install.success" \
      "${thirdpartyPath}/antlr_v"*".download.success" \
      "${thirdpartyPath}/catch2_v"*".install.success" \
      "${thirdpartyPath}/grpc_v"*".install.success" \
      "${thirdpartyPath}/grpc_v"*".download.success" \
      "${thirdpartyPath}/nlohmannjson_v"*".install.success" \
      "${thirdpartyPath}/openBlas_v"*".install.success" \
      "${thirdpartyPath}/openBlas_v"*".download.success" \
      "${thirdpartyPath}/llvm_v"*".install.success" \
      "${thirdpartyPath}/arrow_v"*".install.success" \
      "${thirdpartyPath}/arrow_v"*".download.success" \
      "${llvmCommitFilePath}")

    clean dirs files
}

#******************************************************************************
# Create / Check Indicator-files
#******************************************************************************

#// creates indicator file which indicates successful dependency installation in <projectRoot>/thirdparty/
#// param 1 dependency name
function dependency_install_success() {
    daphne_msg "Successfully installed ${1}."
    touch "${thirdpartyPath}/${1}.install.success"
}
function dependency_download_success() {
    daphne_msg "Successfully downloaded ${1}."
    touch "${thirdpartyPath}/${1}.download.success"
}

#// checks if dependency is installed successfully
#// param 1 dependency name
function is_dependency_installed() {
    [ -e "${thirdpartyPath}/${1}.install.success" ]
}
function is_dependency_downloaded() {
    [ -e "${thirdpartyPath}/${1}.download.success" ]
}

#******************************************************************************
# versions of third party dependencies
#******************************************************************************
antlrVersion=4.9.2
catch2Version=2.13.8
openBlasVersion=0.3.19
abslVersion=20211102.0
grpcVersion=1.38.0
nlohmannjsonVersion=3.10.5
arrowVersion=d9d78946607f36e25e9d812a5cc956bd00ab2bc9

#******************************************************************************
# Set some prefixes, paths and dirs
#******************************************************************************

projectRoot="$(pwd)"
thirdpartyPath="${projectRoot}/thirdparty"

# a convenience indirection to set build/cache/install/source dirs at once (e.g., to /tmp/daphne)
myPrefix=$thirdpartyPath
#myPrefix=/tmp/daphne

daphneBuildDir="$projectRoot/build"
llvmCommitFilePath="${thirdpartyPath}/llvm-last-built-commit.txt"
patchDir="$thirdpartyPath/patches"
installPrefix="${myPrefix}/installed"
buildPrefix="${myPrefix}/build"
sourcePrefix="${myPrefix}/sources"
cacheDir="${myPrefix}/download-cache"

mkdir -p "$cacheDir"

#******************************************************************************
# Parse arguments
#******************************************************************************

# Defaults.
target="daphne"
par_printHelp="0"
par_clean="0"
par_acceptAll="0"
unknown_options=""
BUILD_CUDA="-DUSE_CUDA=OFF"
BUILD_ARROW="-DUSE_ARROW=OFF"
BUILD_DEBUG="-DCMAKE_BUILD_TYPE=Release"

while [[ $# -gt 0 ]]; do
    key=$1
    shift
    case $key in
        -h|--help)
            par_printHelp="1"
            ;;
        --clean)
            par_clean="1"
            ;;
        --cleanAll)
            par_clean="2"
            ;;
        --target)
            target=$1
            shift
            ;;
        -nf|--no-fancy)
            fancy="0"
            ;;
        -y|--yes)
            par_acceptAll="1"
            ;;
        --cuda)
            echo using CUDA
            export BUILD_CUDA="-DUSE_CUDA=ON"
            ;;
        --arrow)
            echo using ARROW
            BUILD_ARROW="-DUSE_ARROW=ON"
            ;;
        --debug)
            echo building DEBUG version
            export BUILD_DEBUG="-DCMAKE_BUILD_TYPE=Debug"
            ;;
        *)
            unknown_options="${unknown_options} ${key}"
            ;;
    esac
done


if [ -n "$unknown_options" ]; then
    printf "Unknown option(s): '%s'\n\n" "$unknown_options"
    printHelp
    exit 1
fi

if [ "$par_printHelp" -eq 1 ]; then
    printHelp
    exit 0
fi

if [ "$par_clean" -eq 1 ]; then
    cleanBuildDirs
    exit 0
fi
if [ "$par_clean" -eq 2 ]; then
    cleanAll
    exit 0
fi


# Print Daphne-Logo when first time executing
if [ ! -d "${projectRoot}/build" ]; then
    printLogo
    sleep 1
fi


# Make sure that the submodule(s) have been updated since the last clone/pull.
# But only if this is a git repo.
if [ -d .git ]; then
    git submodule update --init --recursive
fi


#******************************************************************************
# Download and install third-party material if necessary
#******************************************************************************

#------------------------------------------------------------------------------
# Antlr4 (parser)
#------------------------------------------------------------------------------
antlrJarName="antlr-${antlrVersion}-complete.jar"
antlrCppRuntimeDirName="antlr4-cpp-runtime-${antlrVersion}-source"
antlrCppRuntimeZipName="${antlrCppRuntimeDirName}.zip"

# Download antlr4 C++ run-time if it does not exist yet.
if ! is_dependency_downloaded "antlr_v${antlrVersion}"; then
    daphne_msg "Get Antlr version ${antlrVersion}"
    # Download antlr4 jar if it does not exist yet.
    daphne_msg "Download Antlr v${antlrVersion} java archive"
    wget "https://www.antlr.org/download/${antlrJarName}" -qO "${cacheDir}/${antlrJarName}"
    daphne_msg "Download Antlr v${antlrVersion} Runtime"
    wget https://www.antlr.org/download/${antlrCppRuntimeZipName} -qO "${cacheDir}/${antlrCppRuntimeZipName}"
    rm -rf "${sourcePrefix:?}/$antlrCppRuntimeDirName"
    mkdir --parents "$sourcePrefix/$antlrCppRuntimeDirName"
    unzip -q "$cacheDir/$antlrCppRuntimeZipName" -d "$sourcePrefix/$antlrCppRuntimeDirName"
    dependency_download_success "antlr_v${antlrVersion}"
fi
# build antlr4 C++ run-time
if ! is_dependency_installed "antlr_v${antlrVersion}"; then
    mkdir -p "$installPrefix"/share/antlr4/
    cp "$cacheDir/$antlrJarName" "$installPrefix/share/antlr4/$antlrJarName"
    
    daphne_msg "Applying 0000-antlr-silence-compiler-warnings.patch"
    # disable fail on error as first build might fail and patches might be rejected
    set +e
    patch -Np0 -i "$patchDir/0000-antlr-silence-compiler-warnings.patch" \
      -d "$sourcePrefix/$antlrCppRuntimeDirName"
    # Github disabled the unauthenticated git:// protocol, patch antlr4 to use https://
    # until we upgrade to antlr4-4.9.3+
    sed -i 's#git://github.com#https://github.com#' "$sourcePrefix/$antlrCppRuntimeDirName/runtime/CMakeLists.txt"

    daphne_msg "Build Antlr v${antlrVersion}"
    cmake -S "$sourcePrefix/$antlrCppRuntimeDirName" -B "${buildPrefix}/${antlrCppRuntimeDirName}" \
      -G Ninja -DANTLR4_INSTALL=ON -DCMAKE_INSTALL_PREFIX="$installPrefix" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${buildPrefix}/${antlrCppRuntimeDirName}"
    daphne_msg "Applying 0001-antlr-gtest-silence-warnings.patch"
    patch -Np1 -i "$patchDir/0001-antlr-gtest-silence-warnings.patch" \
      -d "$buildPrefix/$antlrCppRuntimeDirName/runtime/thirdparty/utfcpp/extern/gtest/"

    # enable fail on error again
    set -e
    cmake --build "${buildPrefix}/${antlrCppRuntimeDirName}" --target install

    dependency_install_success "antlr_v${antlrVersion}"
else
    daphne_msg "No need to build Antlr4 again."
fi


#------------------------------------------------------------------------------
# catch2 (unit test framework)
#------------------------------------------------------------------------------
# Download catch2 release zip (if necessary), and unpack the single header file
# (if necessary).
catch2Name="catch2"
catch2ZipName="v$catch2Version.zip"
catch2SingleHeaderInstalledPath=$installPrefix/include/catch.hpp
if ! is_dependency_installed "catch2_v${catch2Version}"; then
    daphne_msg "Get catch2 version ${catch2Version}"
    mkdir --parents "${thirdpartyPath}/${catch2Name}"
    cd "${thirdpartyPath}/${catch2Name}"
    if [ ! -f "$catch2ZipName" ] || [ ! -f "$catch2SingleHeaderInstalledPath" ]
    then
        daphne_msg "Download catch2 version ${catch2Version}"
        wget "https://github.com/catchorg/Catch2/archive/refs/tags/${catch2ZipName}" -qO "${cacheDir}/catch2-${catch2ZipName}"
        unzip -q -p "$cacheDir/$catch2Name-$catch2ZipName" "Catch2-$catch2Version/single_include/catch2/catch.hpp" \
            > "$catch2SingleHeaderInstalledPath"
    fi
    dependency_install_success "catch2_v${catch2Version}"
else
    daphne_msg "No need to download Catch2 again."
fi


#------------------------------------------------------------------------------
# OpenBLAS (basic linear algebra subprograms)
#------------------------------------------------------------------------------
openBlasDirName="OpenBLAS-$openBlasVersion"
openBlasZipName="${openBlasDirName}.zip"
openBlasInstDirName=$installPrefix
if ! is_dependency_downloaded "openBlas_v${openBlasVersion}"; then
    daphne_msg "Get OpenBlas version ${openBlasVersion}"
    wget "https://github.com/xianyi/OpenBLAS/releases/download/v${openBlasVersion}/${openBlasZipName}" \
        -qO "${cacheDir}/${openBlasZipName}"
    unzip -q "$cacheDir/$openBlasZipName" -d "$sourcePrefix"
    dependency_download_success "openBlas_v${openBlasVersion}"
fi
if ! is_dependency_installed "openBlas_v${openBlasVersion}"; then
    cd "$sourcePrefix/$openBlasDirName"
    make -j"$(nproc)"
    make PREFIX="$openBlasInstDirName" install
    cd -
    dependency_install_success "openBlas_v${openBlasVersion}"
else
    daphne_msg "No need to build OpenBlas again."
fi


#------------------------------------------------------------------------------
# nlohmann/json (library for JSON parsing)
#------------------------------------------------------------------------------
nlohmannjsonDirName=nlohmannjson
nlohmannjsonSingleHeaderName=json.hpp
if ! is_dependency_installed "nlohmannjson_v${nlohmannjsonVersion}"; then
    daphne_msg "Get nlohmannjson version ${nlohmannjsonVersion}"
    mkdir -p "${installPrefix}/include/${nlohmannjsonDirName}"
    wget "https://github.com/nlohmann/json/releases/download/v$nlohmannjsonVersion/$nlohmannjsonSingleHeaderName" \
      -qO "${installPrefix}/include/${nlohmannjsonDirName}/${nlohmannjsonSingleHeaderName}"
    dependency_install_success "nlohmannjson_v${nlohmannjsonVersion}"
else
    daphne_msg "No need to download nlohmannjson again."
fi

#------------------------------------------------------------------------------
# abseil (compiled separately to apply a patch)
#------------------------------------------------------------------------------
abslPath=$sourcePrefix/abseil-cpp
if ! is_dependency_downloaded "absl_v${abslVersion}"; then
  daphne_msg "Get abseil version ${abslVersion}"
  rm -rf "$abslPath"
  git clone --depth 1 --branch "$abslVersion" https://github.com/abseil/abseil-cpp.git "$abslPath"
  daphne_msg "Applying 0002-absl-stdmax-params.patch"
  patch -Np1 -i "${patchDir}/0002-absl-stdmax-params.patch" -d "$abslPath"
  dependency_download_success "absl_v${abslVersion}"
fi
if ! is_dependency_installed "absl_v${abslVersion}"; then
    cmake -S "$abslPath" -B "$buildPrefix/absl" -G Ninja -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
      -DCMAKE_INSTALL_PREFIX="$installPrefix" -DCMAKE_CXX_STANDARD=17 -DABSL_PROPAGATE_CXX_STD=ON
    cmake --build "$buildPrefix/absl" --target install
    dependency_install_success "absl_v${abslVersion}"
else
    daphne_msg "No need to build Abseil again."
fi


#------------------------------------------------------------------------------
# gRPC
#------------------------------------------------------------------------------
grpcDirName="grpc"
grpcInstDir=$installPrefix
if ! is_dependency_downloaded "grpc_v${grpcVersion}"; then
    daphne_msg "Get grpc version ${grpcVersion}"
    # Download gRPC source code.
    if [ -d "${sourcePrefix}/${grpcDirName}" ]; then
      rm -rf "${sourcePrefix}/${grpcDirName:?}"
    fi
    git clone -b v$grpcVersion --depth 1 https://github.com/grpc/grpc "$sourcePrefix/$grpcDirName"
    pushd "$sourcePrefix/$grpcDirName"
    git submodule update --init --depth 1 third_party/boringssl-with-bazel
    git submodule update --init --depth 1 third_party/cares/cares
    git submodule update --init --depth 1 third_party/protobuf
    git submodule update --init --depth 1 third_party/re2
    daphne_msg "Applying 0003-protobuf-override.patch"
    patch -Np1 -i "${patchDir}/0003-protobuf-override.patch" -d "$sourcePrefix/$grpcDirName/third_party/protobuf"
    popd
    dependency_download_success "grpc_v${grpcVersion}"
fi
if ! is_dependency_installed "grpc_v${grpcVersion}"; then
    cmake -G Ninja -S "$sourcePrefix/$grpcDirName" -B "$buildPrefix/$grpcDirName" \
      -DCMAKE_INSTALL_PREFIX="$grpcInstDir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_INCLUDE_PATH="$installPrefix/include" \
      -DgRPC_ABSL_PROVIDER=package \
      -DgRPC_ZLIB_PROVIDER=package
    cmake --build "$buildPrefix/$grpcDirName" --target install
    dependency_install_success "grpc_v${grpcVersion}"
else
    daphne_msg "No need to build GRPC again."
fi

#------------------------------------------------------------------------------
# Arrow / Parquet
#------------------------------------------------------------------------------
arrowDirName="arrow"
if [[ "$BUILD_ARROW" == "-DUSE_ARROW=ON" ]]; then
    if ! is_dependency_downloaded "arrow_v${arrowVersion}"; then
        rm -rf ${sourcePrefix}/${arrowDirName}
        git clone -n https://github.com/apache/arrow.git ${sourcePrefix}/${arrowDirName}
        cd ${sourcePrefix}/${arrowDirName}
        git checkout $arrowVersion
        dependency_download_success "arrow_v${arrowVersion}"
    fi
    if ! is_dependency_installed "arrow_v${arrowVersion}"; then
        cmake -G Ninja -S "${sourcePrefix}/${arrowDirName}/cpp" -B "${buildPrefix}/${arrowDirName}" \
            -DCMAKE_INSTALL_PREFIX=${installPrefix} \
            -DARROW_CSV=ON -DARROW_FILESYSTEM=ON -DARROW_PARQUET=ON
        cmake --build "${buildPrefix}/${arrowDirName}" --target install
        dependency_install_success "arrow_v${arrowVersion}"
    else
        daphne_msg "No need to build Arrow again."
    fi
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

llvmName="llvm-project"
llvmCommit="llvmCommit-local-none"
cd "${thirdpartyPath}/${llvmName}"
if [ -e .git ] # Note: .git in the submodule is not a directory.
then
    llvmCommit="$(git log -1 --format=%H)"
fi

if ! is_dependency_installed "llvm_v${llvmCommit}" || [ "$(cat "${llvmCommitFilePath}")" != "$llvmCommit" ]; then
    daphne_msg "Build llvm version ${llvmCommit}"
    cd "${thirdpartyPath}/${llvmName}"
    echo "Need to build MLIR/LLVM."
    cmake -G Ninja -S llvm -B "$buildPrefix/$llvmName" \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=OFF \
       -DLLVM_TARGETS_TO_BUILD="X86" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
       -DLLVM_ENABLE_RTTI=ON \
       -DCMAKE_INSTALL_PREFIX="$installPrefix"
    cmake --build "$buildPrefix/$llvmName" --target check-mlir
    echo "$llvmCommit" > "$llvmCommitFilePath"
    dependency_install_success "llvm_v${llvmCommit}"
else
    daphne_msg "No need to build MLIR/LLVM again."
fi


# *****************************************************************************
# Build DAPHNE target.
# *****************************************************************************

daphne_msg "Build Daphne"

cmake -S "$projectRoot" -B "$daphneBuildDir" -G Ninja $BUILD_CUDA $BUILD_ARROW $BUILD_DEBUG \
  -DCMAKE_PREFIX_PATH="$installPrefix" -DANTLR_VERSION="$antlrVersion"  \
  -DMLIR_DIR="$buildPrefix/$llvmName/lib/cmake/mlir/" \
  -DLLVM_DIR="$buildPrefix/$llvmName/lib/cmake/llvm/"

cmake --build "$daphneBuildDir" --target "$target"

build_ts_end=$(date +%s%N)
daphne_msg "Successfully built Daphne://${target} (took $(printableTimestamp $((build_ts_end - build_ts_begin))))"

set +e