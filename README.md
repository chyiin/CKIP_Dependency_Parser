
# CKIP Dependency Parsing Demo

DEMO Website: https://ckip.iis.sinica.edu.tw/service/dependency-parser/

# Run CKIP tagger server

Requirements (Run Pipfile in ```tmux```)
```
cd tagger
pipenv install
pipenv shell
```
Press button ```ctrl+b+d``` 

# Chinese Quick Parse
INPUT example: ```'我覺得今天天氣真好。'```
```
python3 parser.py
```