#!/bin/bash -ex

# Define the algorithms
ALGORITHMS=("shapleyiq" "ton" "microrank" "microhecl" "microrca")

# Backup original files
cp entrypoint.sh entrypoint.sh.bak
cp info.toml info.toml.bak

# Function to restore original files
restore_files() {
    cp entrypoint.sh.bak entrypoint.sh
    cp info.toml.bak info.toml
}

# Trap to ensure cleanup on exit
trap restore_files EXIT

# Process each algorithm
for algo in "${ALGORITHMS[@]}"; do
    echo "Processing algorithm: $algo"
    
    # Modify entrypoint.sh
    sed -i "s/export ALGORITHM=\${ALGORITHM:-.*}/export ALGORITHM=\${ALGORITHM:-$algo}/" entrypoint.sh
    
    # Modify info.toml
    sed -i "s/name = \".*\"/name = \"$algo\"/" info.toml
    
    # Build and push docker image
    docker build -t "10.10.10.240/library/rca-algo-$algo:latest" .
    docker push "10.10.10.240/library/rca-algo-$algo:latest"
    
    # Upload algorithm
    rca upload-algorithm-harbor ./
    
    echo "Completed processing for $algo"
    echo "----------------------------------------"
done

echo "All algorithms processed successfully!"
