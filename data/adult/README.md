We follow exactly the same data processing procedures described in [the paper](https://arxiv.org/abs/1902.00146) we are comparing with. See ```create_dataset.py``` for the details.

First download raw data:

```
[currently in the under the same directory as this readme file]
cd ./raw_data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
mv adult.data adult.train
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
sed -i '1d' adult.test
cd ../
```
Then go to the `fair_flearn/data/adult` dir, and preprocess data:

```
python create_dataset.py
```

The testing data and training data will be in the ```data/test``` and ```data/train``` folders respectively. These (standard) training and testing samples are exactly the same as those used in the [AFL paper](https://arxiv.org/abs/1902.00146).