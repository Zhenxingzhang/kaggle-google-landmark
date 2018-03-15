#!/usr/bin/env bash

cd /data/landmarks
kg download -u ${username} -p ${password} -c landmark-retrieval-challenge

unzip index.csv.zip

unzip test.csv.zip