# Installation

First install the requirements

```bash
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install -r requirements.txt
```

# Run the evaluation

To run the evaluation execute the following script:
```bash
python pascalvoc.py -gt _annotations.csv -det submission.csv -gtformat xyrb -detformat xyrb
```

You will get the mAP values printed, the one we need is the last one which is mAP.
You can also get it's value from pascalvoc.py in the last lines 
