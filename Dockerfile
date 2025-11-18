FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# 安裝 C++ 編譯工具、CMake、Python（如果之後要用到）
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    git \
    libboost-serialization-dev \
 && rm -rf /var/lib/apt/lists/*

# 工作目錄：專案根目錄（這裡會看到 CMakeLists.txt）
WORKDIR /workspace

# 把整個 repo 複製進來
COPY . .

# 用 CMake 建置：產生 build 目錄並編譯
RUN cmake -S . -B build \
 && cmake --build build -j

# 預設進入 shell，之後你可以手動跑 build 出來的 test binary
CMD ["bash"]
