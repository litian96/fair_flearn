# Fair Resource Allocation in Federated Learning



 

## Preparation

### Download Dependencies

```
pip3 install -r requirements.txt
```

*Please make sure you have all the libraries successfully installed.*

### Generate Datasets

Due to file size constraints, we didn't directly provide datasets in the code. For each dataset used in the paper, see the `README` files in separate `data/$dataset` folders for instructions on preprocessing and/or sampling data.

For example,

under ```fair_flearn/data/fmnist```, we clearly describe how to generate and preprocess the Fashion MNIST dataset.

*[Note: although we create seperate train and test folders for synthetic/vehicle/sent140/shakespeare datasets (which don't have standard testing sets), we run cross validation on different train/test/validation splits of the entire datasets in the code.]*

**In order to run the following demo on the Vehicle dataset, please go to dir `fair_flearn/data/vehicle` and download and generate datasets according to the `README` file under that directory.**

## Start Running

### Example: the Vehicle dataset

*[We provide a quick demo on the Vehicle dataset here. Don't need to change any default parameters in any scripts.]*

First specify GPU ids (we can just use CPUs for Vehicle with linear SVM)

```
export CUDA_VISIBLE_DEVICES=

```
Then go to the `fair_flearn` directory, and start running:

```
bash run.sh $dataset $method $data_partition_seed $q $sampling_device_method | tee $log
```
For Vehicle, `$dataset` is `vehicle`, `$data_partition_seed` can be set to 1, `q` is `0` and `5`. For sampling with weights proportional to the number of data points, `$sampling_device_method` is `2`; for uniform sampling, `$sampling_device_method` is `6`. The exact instructions are as follows.

(1) Experiments to verify the fairness of the q-FFL objective, and compare with uniform sampling schemes:

```
mkdir log_vehicle
bash run.sh vehicle qffedavg 1 0 2 | tee log_vehicle/ffedavg_run1_q0
bash run.sh vehicle qffedavg 1 5 2 | tee log_vehicle/ffedavg_run1_q5
bash run.sh vehicle qffedavg 1 0 6 | tee log_vehicle/fedavg_uniform_run1

```

Plot to re-produce the results in the paper:

(we use `seaborn` to draw the fitting curves of accuracy distributions)

```
pip install seaborn
python plot_fairness.py
```

We could compare the generated `fairness_vehicle.pdf` with Figure 1 (the Vehicle subfigure) and Figure 2 (the Vehicle subfigure) in the paper to validate reproducibility. Note that the accuracy distributions in the paper (both in Figure and Table) are the results averaged across 5 different data partitions with data parititon seeds 1, 2, 3, 4, 5.

(2) For the efficiency experiments:

```
bash run.sh vehicle qffedsgd 1 5 2 | tee log_vehicle/ffedsgd_run1_q5
```

Plot to re-produce the results in the paper:

```
python plot_efficiency.py
```

We could compare the generated `efficiency_qffedavg.pdf` fig with Figure 3 (the Vehicle subfigure) to verify reproducibility.

### Run on other datasets

* First, config `run.sh` based on all hyper-parameters (e.g., batch size, learning rate, etc) reported in the paper (appendix B.2.3). 
* If you would like to run on Sent140, you also need to download a pre-trained embedding file using the following commands (this may take 3-5 minutes):

```
cd fair_flearn/flearn/models/sent140
bash get_embs.sh
```
* We use different models for different datasets, so you might need to change the model name specified by `--model`. The corrsponding model associated with a dataset is described in `fair_flearn/models/$dataset/$model.py`. For instance, if you would like to run on the Shakespeare dataset, you can find the model name under `fair_flearn/models/shakespeare/`, which is `stacked_lstm`, and pass this parameter to `--model='stacked_lstm'`. 
* You also need to specify how many rounds to run using `--num_rounds`. Suggested number of rounds based on our previous experiments are:

```
Vehicle: default
synthetic: 20000
sent140: 200
shakespeare: 80
fashion mnist: 6000
adult: 600
```

For fairness and efficiency experiments, we use four datasets: Vehicle, Sythetic, sent140 and Shakespeare. `method` can be chosen from `[qffedavg, qffedsgd]`. `$sampling` is `2` (with weights of sampling devices proportional to the number of local data points).

```
mkdir log_$dataset
bash run.sh $dataset $method $seed $q $sampling | tee log_$dataset/$method_run$seed_q$q
```

In particular, `$dataset` can be chosen from `[vehicle, synthetic, sent140, shakespeare]`, in accordance with the data directory names under the `fair_flearn/data/` folder.

**Compare with AFL.** We compare wtih AFL using the two datasets (Fashion MNIST and Adult) in the [original paper](https://arxiv.org/abs/1902.00146). 

* Generate data. (data generation process is as described above) 
* Specify parameters. `method` should be specified to be `afl` in order to run AFL algorithms. `data_partition_seed` should be set to 0, such that it won't randomly partition datasets into train/test/validation splits. This allows us to use the same standard public testing set as that in the AFL paper. `track_individual_accuracy` should be set to 1. Here is an example `run.sh` for Adult dataset:

```
python3  -u main.py --dataset=$1 --optimizer=$2  \
            --learning_rate=0.1 \
            --learning_rate_lambda=0.1 \
            --num_rounds=600 \
            --eval_every=1 \
            --clients_per_round=2 \
            --batch_size=10 \
            --q=$4 \
            --model='lr' \
            --sampling=$5  \
            --num_epochs=1 \
            --data_partition_seed=$3 \
            --log_interval=100 \
            --static_step_size=0 \
            --track_individual_accuracy=1 \
            --output="./log_$1/$2_samp$5_run$3_q$4"
```
And then run:

```
bash run.sh adult qffedsgd 0 5 2 | tee log_adult/qffedsgd_q5
bash run.sh adult afl 0 0 2 | tee log_adult/afl
```
* You can find the accuracy numbers in the log files `log_adult/qffedsgd_q5` and `log_adult/afl`, respectively. 


