#!/usr/bin/env bash

# Docker entrypoint script.

set -e

uv sync --locked || true

exec "$@"
