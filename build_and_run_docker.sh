#!/bin/bash
set -euo pipefail

IMAGE_NAME="turbodiff"
CONTAINER_NAME="turbodiff_container"
PORT="8000"
ENV_FILE=".env"
FIREBASE_FILE="firebase-creds.json"

# -------------------------------------------------------------------------
# Force remove old container
# -------------------------------------------------------------------------
echo "Removing old container if it exists..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Give Docker daemon time to fully release container name
sleep 2

# -------------------------------------------------------------------------
# Build the Docker image
# -------------------------------------------------------------------------
echo "Building Docker image: $IMAGE_NAME"
docker buildx build --load -t "$IMAGE_NAME" .

# -------------------------------------------------------------------------
# Run the container with environment and credentials mounted
# -------------------------------------------------------------------------
echo "Starting Docker container: $CONTAINER_NAME"

docker run \
    --init \
    --name "$CONTAINER_NAME" \
    -p "$PORT:$PORT" \
    --env-file "$ENV_FILE" \
    -v "$(pwd)/$FIREBASE_FILE:/app/$FIREBASE_FILE:ro" \
    -d "$IMAGE_NAME"

# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------
echo "✅ Container '$CONTAINER_NAME' is up. Access the API at: http://localhost:$PORT"

# -------------------------------------------------------------------------
# Stream logs so you can see startup output
# -------------------------------------------------------------------------
echo "📜 Streaming container logs (Ctrl+C to stop)..."
docker logs -f --tail 100 "$CONTAINER_NAME"