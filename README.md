
Clone this repo, and install denpendency libs.
```
cd ~
git clone -b gemini2 https://github.com/lbaitemple/apps-md-robots/
cd ~/apps-md-robots
sudo apt-get install -y python3-pyaudio
sudo pip install -r requirements.txt
```

Copy the credential file
```
cd ~/apps-md-robots
mkdir -p ~/.gemini
cp <json_file> ~/.gemini/creds.json
```

Set your google cloud API key in env.example file and then start.
 
```
cd ~/apps-md-robots
cp env.example .env

```

## Run
run app demos, eg. ai_apps
 
```
cd ~/apps-md-robots
python ai_apps/ai_app.py
```

