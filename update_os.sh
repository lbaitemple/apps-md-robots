#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./update_os.sh [--credential /path/to/key.json]

Installs the Ubuntu and Python dependencies, configures PipeWire audio, and
copies a Google Cloud service-account credential to ~/.gemini/creds.json.

The credential defaults to ~/minipupper_creds.json.

Run this script as the normal audio user, not with sudo. The script uses sudo
only to install Ubuntu packages.
EOF
}

if [[ ${EUID} -eq 0 ]]; then
    echo "Error: run this script as the normal audio user, not as root or with sudo." >&2
    exit 1
fi

credential_source="${HOME}/minipupper_creds.json"
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

[[ -f ${credential_source} ]] || { echo "Error: credential file not found: ${credential_source}" >&2; exit 2; }
[[ ! -L ${credential_source} ]] || { echo "Error: credential must not be a symbolic link." >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "${script_dir}"
credential_source=$(realpath -- "${credential_source}")
credential_dir="${HOME}/.gemini"
credential_target="${credential_dir}/creds.json"
bsp_python_module="${HOME}/mini_pupper_bsp/Python_Module"
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
    python3 -m venv --system-site-packages "${venv_dir}"
else
    # The robot hardware drivers are installed system-wide by mini_pupper_bsp.
    python3 -m venv --upgrade --system-site-packages "${venv_dir}"
fi
# Guard against an existing venv retaining an isolated pyvenv.cfg.
if ! grep -Eq '^include-system-site-packages = true$' "${venv_dir}/pyvenv.cfg"; then
    echo "Error: ${venv_dir}/pyvenv.cfg does not enable system site packages." >&2
    exit 1
fi
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install -r "${script_dir}/requirements.txt"
if [[ ! -f ${bsp_python_module}/setup.py ]]; then
    echo "Error: Mini Pupper BSP Python module not found at ${bsp_python_module}." >&2
    exit 1
fi
"${venv_dir}/bin/python" - "${bsp_python_module}" <<'PY'
import pathlib
import site
import sys

bsp_path = pathlib.Path(sys.argv[1]).resolve()
pth_file = pathlib.Path(site.getsitepackages()[0]) / "mini_pupper_bsp.pth"
pth_file.write_text(f"{bsp_path}\n", encoding="utf-8")
PY

install -d -m 700 -- "${credential_dir}"
if [[ ${credential_source} != ${credential_target} ]]; then
    install -m 600 -- "${credential_source}" "${credential_target}"
else
    chmod 600 -- "${credential_target}"
fi

if [[ ! -f ${env_file} && -f ${script_dir}/env.sample ]]; then
    install -m 600 -- "${script_dir}/env.sample" "${env_file}"
fi

audio_runtime_dir="${XDG_RUNTIME_DIR:-/run/user/${UID}}"
audio_bus="unix:path=${audio_runtime_dir}/bus"
pulse_socket="${audio_runtime_dir}/pulse/native"
pulse_server="unix:${pulse_socket}"

# libpulse locates the server through XDG_RUNTIME_DIR. A systemd system unit,
# a sudo shell, and a non-login SSH shell all drop that variable, and libpulse
# then reports "Connection refused" even though every package is installed and
# the server is running. Pinning PULSE_SERVER to the socket path makes the
# lookup independent of how the app is launched.
export XDG_RUNTIME_DIR="${audio_runtime_dir}"
export PULSE_SERVER="${pulse_server}"
if [[ -S ${audio_runtime_dir}/bus ]]; then
    export DBUS_SESSION_BUS_ADDRESS="${audio_bus}"
fi

# Keep the user manager available after reboot for headless robot deployments.
sudo loginctl enable-linger "${USER}"

if [[ ! -S ${audio_runtime_dir}/bus ]]; then
    echo "Error: no user session bus at ${audio_runtime_dir}/bus." >&2
    echo "Neither systemctl --user nor pactl can work without it. Open a real" >&2
    echo "login session as ${USER} (log in locally, or run" >&2
    echo "'sudo machinectl shell ${USER}@'), then re-run this script." >&2
    exit 1
fi

if ! systemctl --user enable --now pipewire.socket pipewire-pulse.socket wireplumber.service; then
    echo "Error: could not start the PipeWire user services." >&2
    systemctl --user --no-pager --lines=20 status pipewire-pulse.service >&2 || true
    exit 1
fi
echo "PipeWire user services enabled."

# pipewire-pulse creates the compatibility socket asynchronously after start.
for _ in $(seq 1 20); do
    [[ -S ${pulse_socket} ]] && break
    sleep 0.25
done
if [[ ! -S ${pulse_socket} ]]; then
    echo "Error: the PulseAudio compatibility socket never appeared at ${pulse_socket}." >&2
    echo "Check 'systemctl --user status pipewire-pulse.service'." >&2
    exit 1
fi

env_tmp=$(mktemp "${script_dir}/.env.tmp.XXXXXX")
chmod 600 -- "${env_tmp}"
if [[ -f ${env_file} ]]; then
    while IFS= read -r line || [[ -n ${line} ]]; do
        case ${line} in
            API_KEY_PATH=*|XDG_RUNTIME_DIR=*|PULSE_SERVER=*) ;;
            *) printf '%s\n' "${line}" >> "${env_tmp}" ;;
        esac
    done < "${env_file}"
fi
# ai_app8.py calls load_dotenv() before it shells out to pactl, and dotenv does
# not overwrite variables that a real session already set, so these entries only
# take effect when the launch context lacks them.
{
    printf 'API_KEY_PATH=%s\n' "${credential_target}"
    printf 'XDG_RUNTIME_DIR=%s\n' "${audio_runtime_dir}"
    printf 'PULSE_SERVER=%s\n' "${pulse_server}"
} >> "${env_tmp}"
mv -- "${env_tmp}" "${env_file}"
env_tmp=""

"${venv_dir}/bin/python" - <<'PY'
import cv2
from api import media_api
print(f"OpenCV {cv2.__version__} and media_api imported successfully")
PY

if ! pactl info >/dev/null 2>&1; then
    echo "Error: the PipeWire services are running, but pactl cannot reach the audio server." >&2
    echo "Audio environment: XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}, PULSE_SERVER=${PULSE_SERVER}" >&2
    echo "Try: systemctl --user restart pipewire.service pipewire-pulse.service wireplumber.service" >&2
    exit 1
fi

# Checking pactl only with this shell's environment gives a false pass: an
# interactive shell has XDG_RUNTIME_DIR, while a systemd unit or a sudo launch
# does not. Verify the stripped environment the app is actually started in.
if env -u XDG_RUNTIME_DIR -u DBUS_SESSION_BUS_ADDRESS \
        PULSE_SERVER="${pulse_server}" pactl info >/dev/null 2>&1; then
    echo "PulseAudio compatibility reachable with and without a session environment."
else
    echo "Error: pactl works in this shell but fails with PULSE_SERVER=${pulse_server}." >&2
    echo "The app would fail the same way when started from a service or with sudo." >&2
    exit 1
fi

echo "Credential installed at ${credential_target} (mode 600)."
echo "Environment updated at ${env_file}."
echo "Setup complete. Run:"
echo "  cd ${script_dir}"
echo "  source .venv/bin/activate"
echo "  python ai_app/ai_app8.py"
