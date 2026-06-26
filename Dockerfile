# Stage 1: Build the C++ library
FROM ubuntu:24.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the source code and build script
COPY include/ ./include/
COPY src/ ./src/
COPY CMakeLists.txt ./

# Build the shared library
RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    make alien_llm_lib

# Stage 2: Final runtime image
FROM ubuntu:24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a virtual environment for Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy built library from builder stage
COPY --from=builder /app/build/libalien_llm_lib.so ./build/libalien_llm_lib.so

# Copy Python application files
COPY requirements.txt .
COPY main.py .
COPY interface.py .
COPY model_checkpoint.bin .
COPY frontend/ ./frontend/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
