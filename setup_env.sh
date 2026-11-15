#!/usr/bin/env bash
# Install Miniconda into ./.miniconda3 and create the tm_flowers env from environment.yml.
# Usage: bash setup_env.sh
# Activate afterwards with: source .miniconda3/bin/activate tm_flowers
set -euo pipefail

# Work relative to this script's folder so it can be run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="$SCRIPT_DIR/.miniconda3"
ENV_NAME="tm_flowers"
CONDA="$CONDA_DIR/bin/conda"

# Some servers export PYTHONPATH / PIP_CONSTRAINT for the system Python,
# which break installs into this env.
unset PYTHONPATH PIP_CONSTRAINT

# 1. Install Miniconda if it is not already there.
if [ ! -x "$CONDA" ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;;
        Linux-aarch64) INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;;
        Darwin-arm64)  INSTALLER="Miniconda3-latest-MacOSX-arm64.sh" ;;
        Darwin-x86_64) INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh" ;;
        *) echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
    esac

    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT
    echo "==> Downloading $INSTALLER"
    curl -fsSL -o "$TMP_DIR/miniconda.sh" "https://repo.anaconda.com/miniconda/$INSTALLER"
    echo "==> Installing Miniconda to $CONDA_DIR"
    bash "$TMP_DIR/miniconda.sh" -b -p "$CONDA_DIR"
else
    echo "==> Miniconda already installed at $CONDA_DIR"
fi

# Run conda through its own Python. If the install path is very long, the installer
# cannot write a normal "#!" line into bin/conda and it would pick up the wrong Python.
conda() { "$CONDA_DIR/bin/python" "$CONDA" "$@"; }

# 2. Use conda-forge only (the Anaconda defaults channel requires accepting its ToS).
CONDARC="$CONDA_DIR/.condarc"
conda config --file "$CONDARC" --remove channels defaults 2>/dev/null || true
conda config --file "$CONDARC" --add channels conda-forge
conda config --file "$CONDARC" --set channel_priority strict

# 3. Create the env, or update it if it already exists.
if [ -d "$CONDA_DIR/envs/$ENV_NAME" ]; then
    echo "==> Updating env $ENV_NAME"
    conda env update -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.yml"
else
    echo "==> Creating env $ENV_NAME"
    conda env create -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.yml"
fi

# 4. Hide PYTHONPATH / PIP_CONSTRAINT while the env is active, restore them on deactivate.
HOOK_DIR="$CONDA_DIR/envs/$ENV_NAME/etc/conda"
mkdir -p "$HOOK_DIR/activate.d" "$HOOK_DIR/deactivate.d"
cat > "$HOOK_DIR/activate.d/clean_container_env.sh" <<'EOF'
# The container exports PYTHONPATH (Python 3.12 packages) and PIP_CONSTRAINT
# (system version pins) that break this Python 3.10 env; hide them while active.
export _TM_OLD_PYTHONPATH="${PYTHONPATH-}"
export _TM_OLD_PIP_CONSTRAINT="${PIP_CONSTRAINT-}"
unset PYTHONPATH PIP_CONSTRAINT
EOF
cat > "$HOOK_DIR/deactivate.d/clean_container_env.sh" <<'EOF'
[ -n "${_TM_OLD_PYTHONPATH-}" ] && export PYTHONPATH="$_TM_OLD_PYTHONPATH"
[ -n "${_TM_OLD_PIP_CONSTRAINT-}" ] && export PIP_CONSTRAINT="$_TM_OLD_PIP_CONSTRAINT"
unset _TM_OLD_PYTHONPATH _TM_OLD_PIP_CONSTRAINT
EOF

# 5. Quick check: TensorFlow version, GPU, and that a model loads.
echo "==> Verifying"
"$CONDA_DIR/envs/$ENV_NAME/bin/python" - <<EOF
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf, cv2
print("tensorflow", tf.__version__, "| opencv", cv2.__version__)
print("GPU:", tf.config.list_physical_devices("GPU") or "none (CPU only)")
model_file = os.path.join("$SCRIPT_DIR", "flowers", "models", "base", "keras_model.h5")
if os.path.exists(model_file):
    from tensorflow.keras.models import load_model
    load_model(model_file, compile=False)
    print("flowers/models/base loads OK")
EOF

echo
echo "Done. Activate with:"
echo "  source \"$CONDA_DIR/bin/activate\" $ENV_NAME"
