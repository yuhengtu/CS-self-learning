### Build/test container ###
# Define builder stage
FROM semiconductor-savants:base AS builder

# Share work directory
COPY . /usr/src/project
WORKDIR /usr/src/project/build

# Build and test
RUN cmake ..
RUN make
RUN ctest --output-on-failure

### Deploy container ###
# Define deploy stage
FROM ubuntu:noble AS deploy

# Share work directory
WORKDIR /app

# Binary from builder
COPY --from=builder /usr/src/project/build/bin/webserver /app/webserver

# Config lives at repo root, so COPY from root into /app
COPY my_config_docker /app/my_config_docker
COPY file/list/ /app/static/
COPY frontend/ /app/frontend/

# Use ENTRYPOINT to specify the binary name
# Update with real server name
ENTRYPOINT ["./webserver"]

# Use CMD to specify arguments to ENTRYPOINT
# Update with real server args
CMD ["my_config_docker"]
