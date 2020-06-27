import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf


from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fair fed maml to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        num_clients = len(self.clients)
        pk = np.ones(num_clients) * 1.0 / num_clients

        train_batches = {}
        for c in self.clients:
            train_batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds + 2)

        test_batches = {}
        for c in self.clients:
            test_batches[c] = gen_batch(c.test_data, self.batch_size, self.num_rounds + 2)

        print('Have generated training and testing batches for all devices/tasks...')

        for i in trange(self.num_rounds + 1, desc='Round: ', ncols=120):

            # only train on non-held-out clients
            indices, selected_clients = self.select_clients(round=i, pk=pk, held_out=self.held_out, num_clients=self.clients_per_round)

            Deltas = []
            hs = []

            selected_clients = selected_clients.tolist()

            for c in selected_clients:
                # communicate the latest model
                c.set_params(self.latest_model)
                weights_before = c.get_params()

                # solve minimization locally
                batch1 = next(train_batches[c])
                batch2 = next(test_batches[c])

                if self.with_maml:
                    _, grads1, loss1 = c.solve_sgd(batch1)
                _, grads2, loss2 = c.solve_sgd(batch2)

                Deltas.append([np.float_power(loss2 + 1e-10, self.q) * grad for grad in grads2[1]])
                hs.append(self.q * np.float_power(loss2+1e-10, (self.q-1)) * norm_grad(grads2[1]) + (1.0/self.learning_rate) * np.float_power(loss2+1e-10, self.q))

            self.latest_model = self.aggregate2(weights_before, Deltas, hs)

        print("###### finish meta-training, start meta-testing ######")


        test_accuracies = []
        initial_accuracies = []
        for c in self.clients[len(self.clients)-self.held_out:]:  # meta-test on the held-out tasks
            # start from the same initial model that is learnt using q-FFL + MAML
            c.set_params(self.latest_model)
            ct, cl, ns = c.test_error_and_loss()
            initial_accuracies.append(ct * 1.0/ns)
            # solve minimization locally
            for iters in range(self.num_fine_tune):  # run k-iterations of sgd
                batch = next(train_batches[c])
                _, grads1, loss1 = c.solve_sgd(batch)
            ct, cl, ns = c.test_error_and_loss()
            test_accuracies.append(ct * 1.0/ns)
        print("initial mean: ", np.mean(np.asarray(initial_accuracies)))
        print("initial variance: ", np.var(np.asarray(initial_accuracies)))
        print(self.output)
        print("personalized mean: ", np.mean(np.asarray(test_accuracies)))
        print("personalized variance: ", np.var(np.asarray(test_accuracies)))
        np.savetxt(self.output+"_"+"test.csv", np.asarray(test_accuracies), delimiter=",")



