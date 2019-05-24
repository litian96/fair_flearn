import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad, norm_grad, norm_grad_sparse

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.q, self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.data_partition_seed, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, data_partition_seed, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u],  data_partition_seed, model) for u, g in zip(users, groups)]
        return all_clients

    def train_error(self):
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.train_error() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        return num_samples, tot_correct


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        return num_samples, tot_correct


    def validate(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.validate()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        return num_samples, tot_correct

    def test_resulting_model(self):
        num_samples = []
        tot_correct = []
        #self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round, pk, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            indices: an array of indices
            self.clients[]
        '''
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round+4)

        

        if self.sampling == 1:
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]
               
        elif self.sampling == 2:
            num_samples = []
            for client in self.clients:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]
        

        elif self.sampling == 3:
            begin_index = round % len(self.clients)
            end_index = (begin_index + 10) % len(self.clients)

            if end_index > begin_index:
                indices = range(begin_index, end_index)
            else:
                indices = list(range(begin_index, len(self.clients))) + list(range(0, end_index))  # python3, must convert to list first, then add
            return indices, np.asarray(self.clients)[indices]
        

        
        #baseline4: square root
        elif self.sampling == 4:
            num_samples = []
            for client in self.clients:
                num_samples.append(client.train_samples)
            total_squared_samples = np.sum(np.sqrt(np.asarray(num_samples)))
            pk = [np.sqrt(item * 1.0) * 1.0 / total_squared_samples for item in num_samples]
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]
             
        # baseline 5/6
        elif self.sampling == 5 or self.sampling == 6:  # sampling=6: uniform selection + simple average
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
            return indices, np.asarray(self.clients)[indices]

        
        # oracle
        else:
            indices = []
            selected_clients=[]
            for i in range(num_clients):
                tmp = np.random.choice(range(int(i*(len(self.clients)/num_clients)),int((i+1)*(len(self.clients)/num_clients))), 1, replace=False) [0]
                indices.append(tmp)
                selected_clients.append(self.clients[tmp])
            return np.asarray(indices), np.asarray(selected_clients)
        
        '''
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]
        '''
        

    def aggregate(self, wsolns): 
        
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])

        for (w, soln) in wsolns:  # w is the number of samples
            
            if self.sampling == 5:
                pass
            elif self.sampling == 4:
                w = np.sqrt(w)*1.0
            else:
                w = 1.0

            total_weight += w 
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln


    def aggregate2(self, weights_before, Deltas, hs): 
        
        demominator = np.sum(np.asarray(hs))
        num_clients = len(Deltas)
        scaled_deltas = []
        for client_delta in Deltas:
            scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

        updates = []
        for i in range(len(Deltas[0])):
            tmp = scaled_deltas[0][i]
            for j in range(1, len(Deltas)):
                tmp += scaled_deltas[j][i]
            updates.append(tmp)

        new_solutions = [(u - v) * 1.0 for u, v in zip(weights_before, updates)]

        return new_solutions

    def heuristic_sgd_update(self, weights_before, soln, loss, q, learning_rate):
        '''
        this is actually normal gradient descent if scaling factor is just learning rate
        '''
        grads = soln[1]
        grad_norm = norm_grad_sparse(grads)  # about 0.1
        q_dynamic = q
        lr = learning_rate

        '''
        scaling_factor = lr * (loss * 1.0 / (loss + (q_dynamic * 1.0 * lr) * grad_norm))
        
        if self.track_loss_grad_ratio:
            print("loss : {}, q/L: {}, (q/L)*g^2: {}".format(loss, q_dynamic * 1.0 * lr, (q_dynamic * 1.0 * lr) * grad_norm))
        if scaling_factor > 1:
            scaling_factor = 1
        '''
        scaling_factor = lr
        scaled_updates = []
        for layer in range(len(grads)):
            if layer == 0:
                indices = grads[0].indices
                values =  grads[0].values
                first_layer_dense = np.zeros((80,8))
                for i in range(indices.shape[0]):
                    first_layer_dense[indices[i], :] = values[i, :]
                scaled_updates.append(scaling_factor * first_layer_dense)
            else:
                scaled_updates.append(scaling_factor * grads[layer])
        new_solutions = [u - v for u, v in zip(weights_before, scaled_updates)]

        return (soln[0], new_solutions)


