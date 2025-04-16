#!/usr/bin/env python

import sys

eval_id = None
for arg in sys.argv[1:]:
    if arg.startswith('--eval='):
        eval_id = arg.split('=')[1]
        break

if not eval_id:
    print("Error: --eval parameter is required")
    sys.exit(1)

with open(f"./evals/{eval_id}/results/aggregate.csv", 'r') as file:
    print(file.read(), end='')
