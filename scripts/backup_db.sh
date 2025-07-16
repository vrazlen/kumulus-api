#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The name of the running database container
DB_CONTAINER="kumulus-api-db-1"
# The database user and name from your docker-compose.yml
DB_USER="kumulus"
DB_NAME="kumulus_db"

# Backup directory
BACKUP_DIR="backups"
# Filename with timestamp
DATE_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FILE_NAME="kumulus_backup_${DATE_STAMP}.sql.gz"

# --- Main Logic ---
echo "--- Starting Database Backup from container '${DB_CONTAINER}' ---"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Use 'docker exec' to run pg_dump inside the running container.
# The output of pg_dump is piped from the container to the host,
# where it is compressed with gzip and saved to a file.
echo "Dumping database '${DB_NAME}' to '${BACKUP_DIR}/${FILE_NAME}'..."
docker exec "$DB_CONTAINER" pg_dump -U "$DB_USER" -d "$DB_NAME" --clean --if-exists | gzip > "${BACKUP_DIR}/${FILE_NAME}"

echo "--- Database Backup Complete: ${BACKUP_DIR}/${FILE_NAME} ---"