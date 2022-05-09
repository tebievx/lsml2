#!/usr/bin/env bash

jupyter lab \
--ip 0.0.0.0 \
--port 4000 \
--no-browser \
--ServerApp.token='' \
--ServerApp.password='' \
--allow-root
