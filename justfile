default:
  just --list

extras := "flash-attn transformer-engine natten"
training_extras := "apex"

# Install inference in existing environment
install cuda='cu126':
    echo {{ cuda }} > .venv/cuda-version
    ./scripts/_sync.sh "{{ extras }}"
    ./.venv/bin/python scripts/test_environment.py

# Install training in existing environment
install-training:
    ./scripts/_sync.sh "{{ extras }} {{ training_extras }}"
    ./.venv/bin/python scripts/test_environment.py --training
