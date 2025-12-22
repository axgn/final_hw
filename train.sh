#!/usr/bin/env bash

kubectl apply -f configs/train.yaml

while true; do
  count=$(find models -maxdepth 1 -type f | wc -l)
  if [ "$count" -gt 1 ]; then
    kubectl apply -f configs/integrajob.yaml
    break
  fi
  sleep 20
done
