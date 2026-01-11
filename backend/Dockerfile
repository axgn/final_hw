FROM drogonframework/drogon:latest AS builder

ENV MYSQL_HOST=mysql
ENV REDIS_HOST=redis

WORKDIR /app

COPY . /app

RUN mkdir -p build \
	&& cd build \
	&& cmake .. -DCMAKE_BUILD_TYPE=Release \
	&& cmake --build . --config Release


FROM drogonframework/drogon:latest

WORKDIR /app

COPY --from=builder /app/build/blog_backend /app/blog_backend

COPY --from=builder /app/build/config.json /app/config.json
COPY sql /app/sql

RUN mkdir -p /app/logs /app/uploads

EXPOSE 8080

WORKDIR /app

CMD ["./blog_backend"]
