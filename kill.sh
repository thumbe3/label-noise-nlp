ps -ef | grep 'train_classifier.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep defunct | grep -v grep | cut -b8-20 | xargs kill -9
