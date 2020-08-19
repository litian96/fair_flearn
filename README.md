# Fair Resource Allocation in Federated Learning


This repository contains the code and experiments for the paper:

> [Fair Resource Allocation in Federated Learning](https://openreview.net/forum?id=ByexElSYDr)
> 
> [ICLR '20](https://iclr.cc/)

 

## Preparation

### Download Dependencies

```
pip3 install -r requirements.txt
```


### Generate Datasets

See the `README` files in separate `data/$dataset` folders for instructions on preprocessing and/or sampling each dataset.

For example,

under ```fair_flearn/data/fmnist```, we clearly describe how to generate and preprocess the Fashion MNIST dataset.


**In order to run the following demo on the Vehicle dataset, please go to `fair_flearn/data/vehicle`, download, and generate the Vehicle dataset following the `README` file under that directory.**

## Get Started

### Example: the Vehicle dataset

*[We provide a quick demo on the Vehicle dataset here. Don't need to change any default parameters in any scripts.]*

First specify GPU ids (we can just use CPUs for Vehicle with a linear SVM)

```
export CUDA_VISIBLE_DEVICES=

```
Then go to the `fair_flearn` directory, and start running:

```
bash run.sh $dataset $method $data_partition_seed $q $sampling_device_method | tee $log
```
For Vehicle, `$dataset` is `vehicle`, `$data_partition_seed` can be set to 1, `q` is `0` for FedAvg, and `5` for q-FedAvg (the proposed objective). For sampling with weights proportional to the number of data points, `$sampling_device_method` is `2`; for uniform sampling (one of the baselines), `$sampling_device_method` is `1`. The exact command lines are as follows.

(1) Experiments to verify the fairness of the q-FFL objective, and compare with uniform sampling schemes:

```
mkdir log_vehicle
bash run.sh vehicle qffedavg 1 0 2 | tee log_vehicle/ffedavg_run1_q0
bash run.sh vehicle qffedavg 1 5 2 | tee log_vehicle/ffedavg_run1_q5
bash run.sh vehicle qffedavg 1 0 1 | tee log_vehicle/fedavg_uniform_run1

```

Plot to re-produce the results in the manuscript:

(we use `seaborn` to draw the fitting curves of accuracy distributions)

```
pip install seaborn
python plot_fairness.py
```

We can then compare the generated `fairness_vehicle.pdf` with Figure 1 (the Vehicle subfigure) and Figure 2 (the Vehicle subfigure) in the paper to validate reproducibility. Note that the accuracy distributions reported (both in figures and tables) are the results averaged across 5 different train/test/validation data partitions with data parititon seeds 1, 2, 3, 4, and 5.

(2) Experiments to demonstrate the communication-efficiency of the proposed method q-FedAvg:

```
bash run.sh vehicle qffedsgd 1 5 2 | tee log_vehicle/ffedsgd_run1_q5
```

Plot to re-produce the results in the paper:

```
python plot_efficiency.py
```

We can then compare the generated `efficiency_qffedavg.pdf` fig with Figure 3 (the Vehicle subfigure) to verify reproducibility.

### Run on other datasets

* First, config `run.sh` based on all hyper-parameters (e.g., batch size, learning rate, etc) reported in the manuscript (appendix B.2.3). 
* If you would like to run on Sent140, you also need to download a pre-trained embedding file using the following commands (this may take 3-5 minutes):

```
cd fair_flearn/flearn/models/sent140
bash get_embs.sh
```
* We use different models for different datasets, so you need to change the model name specified by `--model`. The corrsponding model associated with a dataset is described in `fair_flearn/models/$dataset/$model.py`. For instance, if you would like to run on the Shakespeare dataset, you can find the model name under `fair_flearn/models/shakespeare/`, which is `stacked_lstm`, and pass this parameter to `--model='stacked_lstm'`. 
* You also need to specify total communication rounds using `--num_rounds`. Suggested number of rounds based on our previous experiments are:

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

**Compare with AFL.** We compare wtih the AFL baseline using the two datasets (samplaed Fashion MNIST and Adult) following the [AFL paper](https://arxiv.org/abs/1902.00146). 

* Generate data. (data generation process is as described above) 
* Specify parameters. `method` should be specified to be `afl` in order to run AFL algorithms. `data_partition_seed` should be set to 0, such that it won't randomly partition datasets into train/test/validation splits. This allows us to use the same standard public testing set as that in the AFL paper. `track_individual_accuracy` should be set to 1. Here is an example `run.sh` for the Adult dataset:

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



## References

See our [Fair Federated Learning](https://openreview.net/pdf?id=ByexElSYDr)  manuscript for more details as well as all references.
