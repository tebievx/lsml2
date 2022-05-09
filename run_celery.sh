#!/usr/bin/env bash

celery -A app:celery_app worker --loglevel=INFO -P solo --concurrency=2
