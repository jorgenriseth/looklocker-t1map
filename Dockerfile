FROM rust:alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /build
COPY . .
RUN cargo build --release --locked

FROM scratch
COPY --from=builder /build/target/release/looklocker-t1map /looklocker-t1map
ENTRYPOINT ["/looklocker-t1map"]
