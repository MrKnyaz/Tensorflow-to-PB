import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops


def create_and_save_model():
    # create a simple model
    X = tf.placeholder(tf.float32, shape=(None, 4), name="input")
    W = tf.get_variable("W", [4, 3], initializer=tf.zeros_initializer())  # W parameters with shape [4, 3]
    b = tf.get_variable("b", [3], initializer=tf.zeros_initializer())  # bias
    Z = tf.add(tf.matmul(X, W), b, name="not_activated_output")  # in case we want to check values without activation function
    A = tf.nn.relu(Z, "output")

    # to save model we need session
    with tf.Session() as sess:
        # tensorflow variables have to be initialized
        sess.run(tf.global_variables_initializer())

        # we won't train a model here, we'll just assign some values to variables
        sess.run(tf.assign(W, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))
        sess.run(tf.assign(b, [1, 1.5, 2]))

        # run model to check calculations
        not_activated_result, activated_result = sess.run([Z, A], feed_dict={X: [[4, 3, 2, 1]]})
        print(not_activated_result)  # should be [[41. 51.5 62.]]
        # save model
        save_to_pb(sess)


def save_to_pb(sess):
    output_node_names = "not_activated_output,output"

    # Web have to convert all variables to constants for easy use in other languages,
    # in this case no additional ./variables folder will be created
    # all data will be stored in a single .pb file
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,                                               # We need to pass session object, it contains all variables
        tf.get_default_graph().as_graph_def(),              # also graph definition is necessary
        output_node_names.split(",")                        # we may use multiple nodes for output
    )
    output_graph = "./saved_model.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())       # That's it!


def load_model():
    model_filename = './saved_model.pb'
    with tf.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # tensorflow adds "import/" prefix to all tensors when imports graph definition, ex: "import/input:0"
        # so we explicitly tell tensorflow to use empty string -> name=""
        tf.import_graph_def(graph_def, name="")
    print(tf.get_default_graph().get_operations())  # just print all operations for debug

    # check if model works
    X = tf.get_default_graph().get_tensor_by_name("input:0")
    Z = tf.get_default_graph().get_tensor_by_name("not_activated_output:0")
    with tf.Session() as sess:
        not_activated_result = sess.run([Z], feed_dict={X: [[4, 3, 2, 1]]})
        print(not_activated_result)  # should be [[41. 51.5 62.]]


# create_and_save_model()
load_model()
