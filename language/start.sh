#!/bin/sh
# Build and run the R2‑D2 translator via docker-compose.

set -e

# Build images and start services in the background
docker-compose build
docker-compose up -d
echo "R2‑D2 translator is starting.  API available at http://localhost:8000"
echo "Prometheus available at http://localhost:9090"