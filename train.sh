#!/bin/sh
python trainBRDF.py --cuda --deviceIds 0 --dataRoot dataset/perspective/ --batchSize0 1 --experiment "checkpoints/20210609"
