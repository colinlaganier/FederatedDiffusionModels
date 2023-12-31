#!/bin/bash

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Flower Federated Learning parameters:"
   echo
   echo "Syntax: scriptTemplate [-g|h|v|V]"
   echo "options:"
   echo "c     Number of clients."
   echo "r     Number of rounds."
   echo "e     Number of epochs."
   echo "d     Training dataset path."
   echo "s     Server Address."
   echo "h     Help message."
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

# Default values
server_address="localhost:8080"
num_clients=2
data_path="C:\Users\ColinLaganier\Documents\UCL\Dissertation\Testing\data\cinic-10\federated\5"
num_epochs=1
dataset="emnist"

# Get the options
while getopts c:r:s:d:e:h: flag
do
    case "${flag}" in
        c) num_clients=${OPTARG};;
        r) num_rounds=${OPTARG};;
        s) server_address=${OPTARG};;
        d) data_path=${OPTARG};;
        e) num_epochs=${OPTARG};;
        h) Help
           exit;;
    esac
done

set -e
# --dataset-path $data_path
python server.py  --dataset $dataset --num-clients $num_clients --rounds $num_rounds --epochs $num_epochs --server-address $server_address&
sleep 3  # Sleep for 3s to give the server enough time to start

echo "Starting $num_clients clients."
for ((i = 0; i < $num_clients; i++))
do
    echo "Starting client $i"
    python client.py \
      --cid $i \
      --server-address $server_address \
      --dataset $dataset &
    #   --dataset-path $data_path \
done
echo "Started $num_clients clients."

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
