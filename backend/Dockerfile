## Multi-stage Dockerfile for Drogon blog backend

# Build stage: use official Drogon image with toolchain and headers
FROM drogonframework/drogon:latest AS builder

WORKDIR /app

# Copy backend source and config into the build image
COPY . /app

# Configure and build in Release mode
RUN mkdir -p build \
	&& cd build \
	&& cmake .. -DCMAKE_BUILD_TYPE=Release \
	&& cmake --build . --config Release


# Runtime stage: reuse Drogon base image to ensure all libs are present
FROM drogonframework/drogon:latest

WORKDIR /app

# Copy compiled binary from builder stage
COPY --from=builder /app/build/blog_backend /app/build/blog_backend

# Copy configuration and optional resources
COPY config.json /app/config.json
COPY sql /app/sql

# Ensure logs/uploads directories exist (as used in config.json / app)
RUN mkdir -p /app/logs /app/uploads

# Drogon backend listens on 8080 (see backend/config.json)
EXPOSE 8080

# Run from build directory so relative path ../config.json works
WORKDIR /app/build

CMD ["./blog_backend"]

