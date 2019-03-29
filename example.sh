#!/bin/bash

job_cmd='python -m src.main --epochs 50 --learn_method unsup'

eval $job_cmd
