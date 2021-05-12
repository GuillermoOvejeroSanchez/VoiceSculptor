#!/bin/bash
sudo tee /etc/apt/sources.list.d/rabbitmq.list <<EOF
## Provides modern Erlang/OTP releases
deb https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-erlang/deb/ubuntu focal main
deb-src https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-erlang/deb/ubuntu focal main

## Provides RabbitMQ
deb https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/deb/ubuntu focal main
deb-src https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/deb/ubuntu focal main
EOF
