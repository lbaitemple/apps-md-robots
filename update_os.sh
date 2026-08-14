#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./update_os.sh --credential /absolute/or/relative/path/to/key.json

Installs the Ubuntu and Python dependencies, configures PipeWire audio, and
copies a Google Cloud service-account credential into the project.

Run this script as the normal audio user, not with sudo. The script uses sudo
only to install Ubuntu packages.
EOF
}

if [[ ${EUID} -eq 0 ]]; then
    echo "Error: run this script as the normal audio user, not as root or with sudo." >&2
    exit 1
fi

credential_source=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --credential)
            [[ $# -ge 2 ]] || { echo "Error: --credential needs a file path." >&2; usage; exit 2; }
            credential_source=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

[[ -n ${credential_source} ]] || { echo "Error: --credential is required." >&2; usage; exit 2; }
[[ -f ${credential_source} ]] || { echo "Error: credential file not found: ${credential_source}" >&2; exit 2; }
[[ ! -L ${credential_source} ]] || { echo "Error: credential must not be a symbolic link." >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "${script_dir}"
credential_source=$(realpath -- "${credential_source}")
credential_dir="${script_dir}/.credentials"
credential_target="${credential_dir}/google-cloud.json"
venv_dir="${script_dir}/.venv"
env_file="${script_dir}/.env"
env_tmp=""

cleanup() {
    if [[ -n ${env_tmp} && -f ${env_tmp} ]]; then
        rm -f -- "${env_tmp}"
    fi
}
trap cleanup EXIT

python3 - "${credential_source}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid credential JSON: {exc}")

if not isinstance(data, dict):
    raise SystemExit("Invalid credential JSON: the top-level value must be an object")
required = {"type", "project_id", "private_key", "client_email"}
missing = sorted(required.difference(data))
if missing:
    raise SystemExit("Credential JSON is missing: " + ", ".join(missing))
PY

sudo apt-get update
sudo apt-get install -y \
    build-essential python3-dev python3-venv portaudio19-dev libsndfile1 \
    pipewire pipewire-pulse wireplumber libspa-0.2-modules \
    pulseaudio-utils libasound2-plugins

if [[ ! -x ${venv_dir}/bin/python ]]; then
    python3 -m venv "${venv_dir}"
fi
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install -r "${script_dir}/requirements.txt"

install -d -m 700 -- "${credential_dir}"
if [[ ${credential_source} != ${credential_target} ]]; then
    install -m 600 -- "${credential_source}" "${credential_target}"
else
    chmod 600 -- "${credential_target}"
fi

if [[ ! -f ${env_file} && -f ${script_dir}/env.sample ]]; then
    install -m 600 -- "${script_dir}/env.sample" "${env_file}"
fi

env_tmp=$(mktemp "${script_dir}/.env.tmp.XXXXXX")
chmod 600 -- "${env_tmp}"
if [[ -f ${env_file} ]]; then
    while IFS= read -r line || [[ -n ${line} ]]; do
        [[ ${line} == API_KEY_PATH=* ]] || printf '%s\n' "${line}" >> "${env_tmp}"
    done < "${env_file}"
fi
printf 'API_KEY_PATH=%s\n' "${credential_target}" >> "${env_tmp}"
mv -- "${env_tmp}" "${env_file}"
env_tmp=""

systemctl --user enable pipewire.socket pipewire-pulse.socket wireplumber.service
systemctl --user start pipewire.socket pipewire-pulse.socket wireplumber.service

"${venv_dir}/bin/python" - <<'PY'
import cv2
from api import media_api
print(f"OpenCV {cv2.__version__} and media_api imported successfully")
PY

if pactl info >/dev/null 2>&1; then
    echo "PipeWire PulseAudio compatibility is running."
else
    echo "Warning: packages are installed, but pactl cannot reach the user audio server." >&2
    echo "Log in as ${USER}, then run: systemctl --user start pipewire.socket pipewire-pulse.socket wireplumber.service" >&2
fi

echo "Credential installed at ${credential_target} (mode 600)."
echo "Environment updated at ${env_file}."
echo "Setup complete. Run:"
echo "  cd ${script_dir}"
echo "  source .venv/bin/activate"
echo "  python ai_app/ai_app8.py"
