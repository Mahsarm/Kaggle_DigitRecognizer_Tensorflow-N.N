import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class DigitRecognizer_NeuralNetwork(object):

    def __init__(self, train_path='data/train.csv', test_path='data/test.csv', batch_size=100, epochs=5, layer1_nodes=500,
        layer2_nodes=500, layer3_nodes=500):
        self.df = pd.read_csv(train_path)
        self.submit_data = pd.read_csv(test_path)      
        self.n_classes = 10
        self.input_columns = self.df.shape[1]-1
        self.batch_size = batch_size
        self.epochs = epochs
        self.X = tf.placeholder(tf.float32, shape=(None, self.input_columns))
        self.y = tf.placeholder(tf.float32)
        self.layer1_nodes = layer1_nodes
        self.layer2_nodes = layer2_nodes
        self.layer3_nodes = layer3_nodes
        # keeping the session open to use the session results in different functions
        self.sess = tf.Session()

    def one_hot_digits(self,x):
        res = np.zeros((10,), dtype=float)
        res[x] = 1.0
        return(res)

    def data_slices(self, test_size=0.2):
        pixels_col = [col for col in self.df.columns if col != 'label'] 
        samples = self.df[pixels_col].values
        target = np.apply_along_axis(self.one_hot_digits, axis=1, arr=self.df[['label']])
        self.train_samples, self.validation_samples, self.train_target, self.validation_target = train_test_split(samples, target, test_size=test_size, random_state=4242)

    def NeuralNetwork_model(self):
        hl1_no_nodes = self.layer1_nodes
        hl2_no_nodes = self.layer2_nodes
        hl3_no_nodes = self.layer3_nodes
        self.hidden_layer_one = {'weights': tf.Variable(tf.random_normal(shape = [self.input_columns, hl1_no_nodes])),
                        'biases': tf.Variable(tf.random_normal( shape = [hl1_no_nodes]))}
        self.hidden_layer_two = {'weights': tf.Variable(tf.random_normal(shape = (hl1_no_nodes, hl2_no_nodes))),
                        'biases': tf.Variable(tf.random_normal( shape = [hl2_no_nodes]))}
        self.hidden_layer_three = {'weights': tf.Variable(tf.random_normal(shape = [hl2_no_nodes, hl3_no_nodes])),
                        'biases': tf.Variable(tf.random_normal( shape = [hl3_no_nodes]))}
        self.output_layer = {'weights': tf.Variable(tf.random_normal(shape = [hl3_no_nodes, self.n_classes])),
                    'biases': tf.Variable(tf.random_normal( shape = [self.n_classes]))}
        layer1_output = tf.add(tf.matmul(self.X, self.hidden_layer_one['weights']) , self.hidden_layer_one['biases'])
        layer1_output = tf.nn.relu(layer1_output)

        layer2_output = tf.add(tf.matmul(layer1_output , self.hidden_layer_two['weights']) , self.hidden_layer_two['biases'])
        layer2_output = tf.nn.relu(layer2_output)

        layer3_output = tf.add(tf.matmul(layer2_output, self.hidden_layer_three['weights']) , self.hidden_layer_three['biases'])
        layer3_output = tf.nn.relu(layer3_output)

        output = tf.matmul(layer3_output, self.output_layer['weights']) + self.output_layer['biases']
        return output
       

    def train_NeuralNetwork(self, X_test = None, y_test = None):
        X_test = X_test if X_test is not None else self.validation_samples
        y_test = y_test if y_test is not None else self.validation_target
        prediction = self.NeuralNetwork_model()

        beta = 0.01
        no_of_batches=self.train_samples.shape[0]/self.batch_size
        regularizers =(tf.nn.l2_loss( self.hidden_layer_one['weights']) +  tf.nn.l2_loss(self.hidden_layer_two['weights']) + 
            tf.nn.l2_loss( self.hidden_layer_three['weights'] + tf.nn.l2_loss( self.output_layer['weights'])))                   
        cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = self.y)) + beta * regularizers)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        is_correct = tf.equal(tf.argmax(prediction,1) , tf.argmax(self.y,1))    
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        self.sess.run(tf.global_variables_initializer())
  
        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            X_train, y_train = shuffle(self.train_samples, self.train_target, random_state=45)

            for _ in range(no_of_batches):
                train_feed_dict = {self.X: X_train[(_*self.batch_size):((_+1)*self.batch_size)], self.y:y_train[(_*self.batch_size):((_+1)*self.batch_size)]}
                _, c = self.sess.run([optimizer, cost], train_feed_dict)
                epoch_loss += c

            test_feed_dict = {self.X:X_test, self.y:y_test}     
            test_accuracy = 100*self.sess.run(accuracy,feed_dict=test_feed_dict)
            print('Epoch', epoch, 'completed out of', self.epochs, 'loss:', epoch_loss, 'Accuracy:', test_accuracy)
        self.prediction = prediction

    def predict(self, X_test=None):
        X_test = X_test if X_test is not None else self.submit_data.values
        test_predictions = self.sess.run(tf.argmax(self.prediction,1), feed_dict={self.X: X_test})
        columns = ['ImageId','Label']
        sub_df = pd.DataFrame(columns=columns)
        sub_df['Label'] = test_predictions
        sub_df['ImageId'] = sub_df.index + 1
        sub_file = sub_df.to_csv('submission.csv', index=False)
        submission_set = pd.read_csv('submission.csv' , nrows= 10)
        print(submission_set)


digit_neuralnetwork = DigitRecognizer_NeuralNetwork(epochs=100)
digit_neuralnetwork.data_slices()
digit_neuralnetwork.train_NeuralNetwork()


