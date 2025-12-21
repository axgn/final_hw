## Multi-stage Dockerfile for Drogon blog backend

# Build stage: use official Drogon image with toolchain and headers
FROM drogonframework/drogon:latest AS builder

# MySQL and Redis IPs can be passed as build args, then exposed as envs
ENV MYSQL_HOST=mysql
ENV REDIS_HOST=redis

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

# Re-declare build args and export them as runtime environment variables

WORKDIR /app

# Copy compiled binary from builder stage
COPY --from=builder /app/build/blog_backend /app/blog_backend

# Copy generated configuration and optional resources
COPY --from=builder /app/build/config.json /app/config.json
COPY sql /app/sql

# Ensure logs/uploads directories exist (as used in config.json / app)
RUN mkdir -p /app/logs /app/uploads

# Drogon backend listens on 8080 (see backend/config.json)
EXPOSE 8080

# Run from build directory so relative path ../config.json works
WORKDIR /app

CMD ["./blog_backend"]
