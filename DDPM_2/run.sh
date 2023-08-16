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
server_address="[::]:8080"
num_clients=2
data_path="C:\Users\ColinLaganier\Documents\UCL\Dissertation\Testing\data\cinic-10\federated\5"
num_epochs=1
dev=0
device_count=(0 0 0 0)

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

python server.py --dataset-path $data_path --num-clients $num_clients --rounds $num_rounds --epochs $num_epochs --device $device&
sleep 3  # Sleep for 3s to give the server enough time to start
# Increment device count
((device_count[$dev]++))
((dev++))

echo "Starting $num_clients clients."
for ((i = 0; i < $num_clients; i++))
do
    if [[ $((device_count[$dev])) == 2 ]]
    then
        ((dev++))
    fi
    if [[ $dev == 4 ]]
    then
        dev=0
    fi 
    echo "Starting client $i"
    python client.py \
      --dataset-path $data_path \
      --cid $i \
      --device $device &
    #   --server_address=$SERVER_ADDRESS &
    device_count = $((device_count + 1))
    ((device_count[$dev]++))
    ((dev++))
done
echo "Started $num_clients clients."

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
sh run.sh c /home/ec2-user/FedKDD/dataset/cinic-10/5
sh run.sh -c 2 -r 1 -e 1 -d /home/ec2-user/FedKDD/dataset/cinic-10/5