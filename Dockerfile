FROM rust as builder
WORKDIR /usr/src/app

# Install dependencies
# Libopenslide & libclang 
RUN apt update && \
    apt install -y libopenslide-dev \
    libclang-dev

# Libtorch 
RUN wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip && \
 unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu117.zip && \
 rm libtorch-cxx11-abi-shared-with-deps-2.0.1+cu117.zip 
ENV LIBTORCH="/usr/src/app/libtorch"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${LIBTORCH}/lib"
# Cargo based dependencies (we use a dummy main.rs to build the dependencies to benefit from docker's caching)
COPY Cargo.toml Cargo.lock .cargo/ ./
RUN mkdir src/ && \
 echo "fn main() {}" > src/dummy.rs && \
 sed -i 's# \[\[dummy placeholder\]\]#\[\[dummy placeholder\]\]\n[[bin]]\nname = "dummy"\npath = "src/dummy.rs"#g' Cargo.toml && \
 cargo build --release --bin="dummy"

# Build the actual application
COPY src/ src/
RUN cargo build --release

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS cu121
WORKDIR /app
VOLUME [ "/in-slides", "/in-geojson", "/out" ]

# Install dependencies (we reuse the ones from the builder)
COPY --from=builder /usr/src/app/libtorch libtorch
RUN ls /app/libtorch/**
ENV LIBTORCH="/app/libtorch"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${LIBTORCH}/lib"
RUN cat /etc/os-release
RUN apt update -y && \
    apt install -y libopenslide0 \
       libgomp1

# Copy the actual application
COPY --from=builder /usr/src/app/target/release/nuclei-feature-extraction .
COPY run.sh .
RUN chmod +x run.sh

# Run the application
ENTRYPOINT [ "./run.sh", "/in-geojson", "/in-slides", "/out"]


FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04 AS cu113
WORKDIR /app
VOLUME [ "/in-slides", "/in-geojson", "/out" ]

# Install dependencies (we reuse the ones from the builder)
COPY --from=builder /usr/src/app/libtorch libtorch
RUN ls /app/libtorch/**
ENV LIBTORCH="/app/libtorch"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${LIBTORCH}/lib"
RUN cat /etc/os-release
RUN apt update -y && \
    apt install -y libopenslide0 \
       libgomp1

# Copy the actual application
COPY --from=builder /usr/src/app/target/release/nuclei-feature-extraction .
COPY run.sh .
RUN chmod +x run.sh

# Run the application
ENTRYPOINT [ "./run.sh", "/in-geojson", "/in-slides", "/out"]
