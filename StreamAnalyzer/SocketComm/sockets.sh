#!/bin/bash

python3 dash_try.py &
python3 main_client.py &

read -r

echo "Stopping Dash Server and Voice Client..."

pkill -f dash_try.py
pkill -f main_client.py
