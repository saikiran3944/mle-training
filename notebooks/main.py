import subprocess

# from ../src import ingest_data, score, train

subprocess.run(["python", "../src/ingest_data.py"])
subprocess.run(["python", "../src/train.py"])
subprocess.run(["python", "../src/score.py"])
