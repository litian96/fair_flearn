import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, project


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using agnostic flearn (non-stochastic version) to Train')
        self.inner_opt = tf.train.AdagradOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.latest_lambdas = np.ones(len(self.clients)) * 1.0 / len(self.clients)
        self.resulting_model = self.client_model.get_params()  # this is only for the agnostic flearn paper

    def train(self):

        print('Training with {} workers ---'.format(self.clients_per_round))
        num_clients = len(self.clients)
        pk = np.ones(num_clients) * 1.0 / num_clients

        batches = {}
        for c in self.clients:
            batches[c] = gen_epoch(c.train_data, self.num_rounds+2)

        for i in trange(self.num_rounds+1, desc='Round: ', ncols=120):
            # test model
            if i % self.eval_every == 0:               
                self.client_model.set_params(self.resulting_model)
                stats = self.test_resulting_model()                   
                tqdm.write('At round {} testing accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))
                test_accuracies = np.divide(np.asarray(stats[3]), np.asarray(stats[2]))
                for idx in range(len(self.clients)):
                    tqdm.write('Client {} testing accuracy: {}'.format(self.clients[idx].id, test_accuracies[idx]))
                  
            solns = []
            losses = []
            for idx, c in enumerate(self.clients):
                c.set_params(self.latest_model)

                batch = next(batches[c])
                _, grads, loss = c.solve_sgd(batch) # this gradient is with respect to w

                losses.append(loss)
                solns.append((self.latest_lambdas[idx],grads[1]))
            
            avg_gradient = self.aggregate(solns)

            for v,g in zip(self.latest_model, avg_gradient):
                v -= self.learning_rate * g
            
            for idx in range(len(self.latest_lambdas)):
                self.latest_lambdas[idx] += self.learning_rate_lambda * losses[idx]

            self.latest_lambdas = project(self.latest_lambdas)

            for k in range(len(self.resulting_model)):
                self.resulting_model[k] = (self.resulting_model[k] * i + self.latest_model[k]) * 1.0 / (i+1)

