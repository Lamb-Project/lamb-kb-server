# Build stage
FROM node:20-bookworm AS builder

# Install build dependencies including Python, build tools, and Rust
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for @mendable/firecrawl-rs)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# ARG for Firecrawl version
ARG VERSION=main
RUN if [ "$VERSION" = "latest" ]; then \
      git clone https://github.com/mendableai/firecrawl.git .; \
    else \
      git clone --branch ${VERSION} https://github.com/mendableai/firecrawl.git .; \
    fi

# Navigate to the API app directory
WORKDIR /build/apps/api

# Install pnpm globally
RUN npm install -g pnpm@9

# Install dependencies (use --frozen-lockfile for reproducibility)
RUN pnpm install --frozen-lockfile

# Build the app
RUN pnpm build

# Prune dev dependencies for production
RUN pnpm prune --production

# Final runtime stage
FROM node:20-bookworm-slim

# Install only runtime dependencies for the final image
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy built artifacts from builder stage
COPY --from=builder /build/apps/api/dist ./dist
COPY --from=builder /build/apps/api/node_modules ./node_modules
COPY --from=builder /build/apps/api/package.json ./package.json
COPY --from=builder /build/apps/api/.env.example ./.env

# Install pnpm in the runtime image
RUN npm install -g pnpm@9

# Expose the port
EXPOSE 3002

# Run the app
CMD ["pnpm", "start"]