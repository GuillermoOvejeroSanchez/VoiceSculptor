#!/bin/bash

python3 dash_try.py &
python3 main_client.py &

read

echo "Stopping Dash Server and Voice Client..."

pkill -f main_client && pkill -f dash_try
