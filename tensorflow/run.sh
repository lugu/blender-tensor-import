#!/bin/sh

set -x -u -e

# create directory for the experience
DIR=$(mktemp -d "data.XXXXXXXXXX")
# allocate one random port for the web server
PORT=$(( ( RANDOM % 1000 ) + 6006 ))

# measure trainning duration
time ./mnist.py "$DIR"

# plot results
tensorboard "--logdir=train:$DIR/train,test:$DIR/test" --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1 &
sleep 1
surf "http://127.0.0.1:$PORT"
