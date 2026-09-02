# MD Robot Starter Kits AI Demos
[MangDang](https://www.mangdang.net/) Online channel: [Discord](https://discord.gg/xJdt3dHBVw), [FaceBook](https://www.facebook.com/groups/716473723088464), [YouTube](https://www.youtube.com/channel/UCqHWYGXmnoO7VWHmENje3ug/featured), [Twitter](https://twitter.com/LeggedRobot)

MD Robot Starter Kits: Unlock your AI Dream Job.
Make robotics easier for schools, homeschool families, enthusiasts and beyond.

- Generative AI: Support ChatGPT, Gemini and Claude
- ROS: support ROS2(Humble) SLAM & Navigation robot dog at low-cost price
- OpenCV: support OpenCV official OAK-D-Lite 3D camera module and single MIPI camera
- Open-source: DIY and customize what you want.
- Raspberry Pi: it’s super expandable, endorsed by Raspberry Pi.

## Overview

The AI applications can be run on MD legged Robot Kits. 
The default branch works on Mini Pupper2(G), please click the picture and refer to the demo video.

[![Run on MD-Puppy1](https://img.youtube.com/vi/mIDuIZCevIg/0.jpg)](https://www.youtube.com/watch?v=mIDuIZCevIg)

The new branch for Mini Pupper will be added soon, please click the picture and refer to the demo video.

[![Run on MD-Puppy1](https://img.youtube.com/vi/bvH-lA1IHig/0.jpg)](https://www.youtube.com/watch?v=bvH-lA1IHig)

## Preparation

Please make sure Mini Pupper 2(G) can walk first. 

- Download and flash the [pre-built base image file (like v2_stanford*.img) ](https://drive.google.com/drive/folders/1ZF4vulHbXvVF4RPWWGxEe7rxcJ9LyeEu?usp=sharing), or 
- Build the base environment by yourself. 

Step 1: Install the [BSP repo](https://github.com/mangdangroboticsclub/mini_pupper_2_bsp)

Step 2: Install the [quadruped repo](https://github.com/mangdangroboticsclub/StanfordQuadruped )


## Install

For the video guide, please click the picture and refer to the demo video.

[![Installation Guide](https://img.youtube.com/vi/1AkhJi2o8rM/0.jpg)](https://www.youtube.com/watch?v=1AkhJi2o8rM)


Clone this repo and install the system dependencies. Full-duplex barge-in on
Ubuntu 24.04 uses PipeWire's PulseAudio compatibility layer, the ALSA Pulse
plugin, and a logged-in user audio session.

For an automated installation, provide the downloaded Google Cloud service
account JSON to `update_os.sh`. Run the script as the normal audio user; it asks
for `sudo` only when installing OS packages. It validates and copies the key to
`.credentials/google-cloud.json`, restricts it to the current user, and updates
`.env` automatically. It also enables the user's persistent systemd session and
starts PipeWire, PipeWire-Pulse, and WirePlumber.

```bash
cd ~/apps-md-robots
chmod +x update_os.sh
./update_os.sh --credential ~/Downloads/google-cloud-key.json
```

The equivalent manual installation follows.

```bash
cd ~
git clone -b physicalAI https://github.com/mangdangroboticsclub/apps-md-robots
cd apps-md-robots
sudo apt-get update
sudo apt-get install -y \
    build-essential python3-dev python3-venv portaudio19-dev libsndfile1 \
    pipewire pipewire-pulse wireplumber libspa-0.2-modules \
    pulseaudio-utils libasound2-plugins
```

Install the Python packages in a virtual environment. Do not use `sudo pip` on
Ubuntu 24.04. The requirements deliberately use only
`opencv-python-headless`; installing another OpenCV wheel alongside it can make
`cv2` require the unavailable desktop library `libGL.so.1`.

```bash
cd ~/apps-md-robots
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned `langchain-google-vertexai==2.0.23` release requires
`google-cloud-storage<3`. The `google-cloud-aiplatform` notice about future
support for storage 2.x is therefore expected and does not stop the app. Do not
upgrade storage to 3.x independently; update the LangChain integration and test
the application at the same time.

Start and verify the per-user audio services. Run these commands as the normal
logged-in desktop/audio user, without `sudo`:

```bash
systemctl --user enable --now pipewire.socket pipewire-pulse.socket wireplumber.service
systemctl --user --no-pager status pipewire.service pipewire-pulse.service wireplumber.service
pactl info
```

`pactl info` must report a server before full-duplex barge-in can start. If it
prints `Connection refused`, confirm that a user session bus exists:

```bash
printf 'XDG_RUNTIME_DIR=%s\nDBUS_SESSION_BUS_ADDRESS=%s\n' \
    "$XDG_RUNTIME_DIR" "$DBUS_SESSION_BUS_ADDRESS"
loginctl user-status "$USER"
```

Both environment values should be populated. Log in locally as the audio user
and run the app from that session. Do not launch it with `sudo`; a root or
system-service process normally cannot access the user's PipeWire server.

`update_os.sh` enables the persistent user session automatically. If it warns
that it could not start PipeWire, log out and back in as the audio user, then
restart the services and verify the PulseAudio compatibility server:

```bash
systemctl --user restart pipewire.service pipewire-pulse.service wireplumber.service
pactl info
```

For a manual setup, copy the supplied credential to the standard Gemini
location and create `.env` from the sample:
 
```bash
mkdir -p ~/.gemini
cp ~/minipupper_creds.json ~/.gemini/creds.json
cp env.sample .env
chmod 700 ~/.gemini
chmod 600 ~/.gemini/creds.json
chmod 600 .env
```

## Run

### if running in jupyter service

add the following into `/etc/systemd/system/jupyterlab.service `

```
[Service]
Environment="PULSE_SERVER=unix:/run/user/1000/pulse/native"
Environment="DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus"
```

Activate the same virtual environment and run the desired demo as the logged-in
audio user. For example:

```bash
cd ~/apps-md-robots
source .venv/bin/activate
python ai_app/ai_app8.py
```
