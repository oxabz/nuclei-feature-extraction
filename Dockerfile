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
 echo "fn main() {}" > src/main.rs && \
 cargo build --release && \
 rm -rf src/

# Build the actual application
COPY src/ src/
RUN cargo build --release

FROM nvidia/cuda:12.1.1-runtime-ubi8 
WORKDIR /usr/src/app
VOLUME [ "/in-slides", "/in-geojson", "/out" ]

# Install dependencies (we reuse the ones from the builder)
COPY --from=builder /usr/src/app/libtorch libtorch
ENV LIBTORCH="/usr/src/app/libtorch"

# Copy the actual application
COPY --from=builder /usr/src/app/target/release/nuclei-feature-extraction .
COPY run.sh .
RUN chmod +x run.sh 

# Run the application
CMD [ "./run.sh", "/in-slides", "/in-geojson", "/out", "$EXTRACTION_OUT_EXT" ]