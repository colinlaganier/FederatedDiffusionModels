#!/bin/bash

set -e

SERVER_ADDRESS="[::]:8080"
NUM_CLIENTS=2
DATA_PATH="C:\Users\ColinLaganier\Documents\UCL\Dissertation\Testing\data\cinic-10\federated\5"

python server.py --dataset-path $DATA_PATH &
sleep 3  # Sleep for 3s to give the server enough time to start

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client $i"
    python client.py \
      --dataset-path $DATA_PATH \
      --cid $i &
    #   --server_address=$SERVER_ADDRESS &
done
echo "Started $NUM_CLIENTS clients."

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait