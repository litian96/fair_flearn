import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        """
        Initialize the model.

        Args:
            self: (todo): write your description
            params: (dict): write your description
            learner: (todo): write your description
            dataset: (todo): write your description
        """
        print('Using fair fed avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        """
        Training function.

        Args:
            self: (todo): write your description
        """
        print('Training with {} workers ---'.format(self.clients_per_round))

        num_clients = len(self.clients)
        pk = np.ones(num_clients) * 1.0 / num_clients

        for i in range(self.num_rounds+1):
            if i % self.eval_every == 0:
                num_test, num_correct_test = self.test() # have set the latest model for all clients
                num_train, num_correct_train = self.train_error()  
                num_val, num_correct_val = self.validate()  
                tqdm.write('At round {} testing accuracy: {}'.format(i, np.sum(np.array(num_correct_test)) * 1.0 / np.sum(np.array(num_test))))
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(np.array(num_correct_train)) * 1.0 / np.sum(np.array(num_train))))
                tqdm.write('At round {} validating accuracy: {}'.format(i, np.sum(np.array(num_correct_val)) * 1.0 / np.sum(np.array(num_val))))
            
            
            if i % self.log_interval == 0 and i > int(self.num_rounds/2):                
                test_accuracies = np.divide(np.asarray(num_correct_test), np.asarray(num_test))
                np.savetxt(self.output + "_" + str(i) + "_test.csv", test_accuracies, delimiter=",")
                train_accuracies = np.divide(np.asarray(num_correct_train), np.asarray(num_train))
                np.savetxt(self.output + "_" + str(i) + "_train.csv", train_accuracies, delimiter=",")
                validation_accuracies = np.divide(np.asarray(num_correct_val), np.asarray(num_val))
                np.savetxt(self.output + "_" + str(i) + "_validation.csv", validation_accuracies, delimiter=",")
            
            
            indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)

            Deltas = []
            hs = []

            selected_clients = selected_clients.tolist()

            for c in selected_clients:                
                # communicate the latest model
                c.set_params(self.latest_model)
                weights_before = c.get_params()
                loss = c.get_loss() # compute loss on the whole training data, with respect to the starting point (the global model)
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                new_weights = soln[1]

                # plug in the weight updates into the gradient
                grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]
                
                Deltas.append([np.float_power(loss+1e-10, self.q) * grad for grad in grads])
                
                # estimation of the local Lipchitz constant
                hs.append(self.q * np.float_power(loss+1e-10, (self.q-1)) * norm_grad(grads) + (1.0/self.learning_rate) * np.float_power(loss+1e-10, self.q))

            # aggregate using the dynamic step-size
            self.latest_model = self.aggregate2(weights_before, Deltas, hs)

                    



