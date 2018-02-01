from graph_constructor_Yazabi import *

# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []
#matplotlib inline

def train_neural_network():
    import tensorflow as tf 
    x = tf.placeholder('float', [None, len(train_x.columns)])
    y = tf.placeholder('float')
    lr = 0.25
    dropout_keep_prob = tf.placeholder(tf.float32)
    prediction = neural_network_model(x, dropout_keep_prob)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost) #AdamOptimizer
    
    correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    
    iteration = 1000
    for epoch in range(iteration):
        loss = 0
        _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict = {x: train_x, y: train_y, dropout_keep_prob: 0.9})
        loss += c
        train_losses.append(c)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training: 
        if (epoch % 100 == 0 and epoch != 0):
            
            # To not spam console, show training accuracy/loss in this "if"
            print('Epoch', epoch, 'completed out of', iteration, ":\n", 
                  "   Loss = " , "{:.6f}".format(c) , \
                  ", Accuracy = {}".format(acc))
            
            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            _c, _acc = sess.run(
                [cost, accuracy], 
                feed_dict={
                    x: test_x,
                    y: test_y,
                    dropout_keep_prob: 1.
                }
            )
            test_losses.append(_c)
            test_accuracies.append(_acc)
            print("PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(_c) + \
                  ", Accuracy = {}".format(_acc))

    print("Optimization Finished!")
    
    # Accuracy for test data
    
    one_hot_predictions, accuracy_, final_loss = sess.run(
        [prediction, accuracy, cost],
        feed_dict={
            x: test_x,
            y: test_y, 
            dropout_keep_prob: 1.
        }
    )

    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    
    indep_train_axis = np.array(range(iteration)) 
    plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")
    
    indep_test_axis = (np.array(range(100, iteration,100)))
#        
    plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")
    
    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')
    
    plt.show()
#*******************************************************
    from sklearn import metrics
    predictions = one_hot_predictions.argmax(1)
    
    print("Testing Accuracy: {}%".format(100*accuracy_))
    
    print("")
    print("Precision: {}%".format(100*metrics.precision_score(Y_test_numeric, predictions, average="weighted")))
    print("Recall: {}%".format(100*metrics.recall_score(Y_test_numeric, predictions, average="weighted")))
    print("f1_score: {}%".format(100*metrics.f1_score(Y_test_numeric, predictions, average="weighted")))
    
    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(Y_test_numeric, predictions)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
    
    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)
    print("Note: training and testing data is not equally distributed amongst classes, ")
    print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")
#    
    # Plot Results: 
    
    cm = normalised_confusion_matrix
    import itertools
    cm = normalised_confusion_matrix
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    
    data = cm 
    
    width = 12
    height = 12
    LABELS = list(n for _, n in activity_map)
    
    fig, ax = plt.subplots()
    im = ax.pcolor(data, cmap='viridis', edgecolor='black', linestyle=':', lw=1)
    fig.colorbar(im)
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set(ticks=np.arange(0.5, len(LABELS)), ticklabels=LABELS)
    
    tick_marks = np.arange(n_classes) 
    ax.set_xticklabels(LABELS, rotation = 90)
    ax.set_yticklabels(LABELS)  
    
    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y].round(1)), xy=(y+0.5, x+0.5), 
                        horizontalalignment='center',
                        verticalalignment='center')
             
    plt.show()
    sess.close()
    pass


def main():
    train_neural_network()
    
if __name__ == '__main__':
     main()