#!/usr/bin/env bash

mlflow server \
--backend-store-uri="sqlite:////project/mlflow/mlflow.db" \
--default-artifact-root="/project/mlflow/artifacts" \
--host 0.0.0.0 \
--port 5000 \
--workers 2
