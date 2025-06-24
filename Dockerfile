FROM golang:1.21

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

ENV CGO_ENABLED=1
ENV GOOS=linux
ENV GOARCH=amd64

RUN cd bench && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

RUN cd bench && go build -o bench main.go

ENTRYPOINT ["bash", "bench/run.sh"] 