#!/bin/bash

echo "Starting Password Hashing Performance Analysis System..."

mkdir -p data/db
mkdir -p results

echo "Cleaning up any existing MongoDB processes..."
pkill -9 mongod 2>/dev/null || true
rm -f data/db/mongod.lock 2>/dev/null || true
sleep 1

echo "Starting MongoDB..."
mongod --dbpath data/db --bind_ip 127.0.0.1 --port 27017 --logpath data/mongodb.log --fork

echo "Waiting for MongoDB to be ready..."
sleep 3
echo "✓ MongoDB is ready!"

echo "Starting Flask application..."
python app.py
