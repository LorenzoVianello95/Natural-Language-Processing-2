import tensorflow as tf
import pickle
import numpy as np
import tqdm
import sys
import math

from data_processing import create_padding,divide_dev,split_sentences

#RESTORE ALL THE PREPROCESSED DATA
f = open('embedding.pckl', 'rb')
embeddings = pickle.load(f)
f.close()
f = open('X.pckl', 'rb')
X = pickle.load(f)
f.close()
f = open('Y.pckl', 'rb')
Y = pickle.load(f)
f.close()
f = open('lentgh_of_sentences.pckl', 'rb')
lentgh_of_sentences = pickle.load(f)
f.close()
f = open('array_of_elements.pckl', 'rb')
[NUM_SENTECES,ntags] = pickle.load(f)
f.close()
#dev:
f = open('X_dev.pckl', 'rb')
X_dev = pickle.load(f)
f.close()
f = open('Y_dev.pckl', 'rb')
Y_dev = pickle.load(f)
f.close()
f = open('lentgh_of_sentences_dev.pckl', 'rb')
lentgh_of_sentences_dev = pickle.load(f)
f.close()
f = open('datasets_dev.pckl', 'rb')
datasets_dev = pickle.load(f)
f.close()

max_len_sent=max(max(lentgh_of_sentences),max(lentgh_of_sentences_dev))
print("max len sentences",max_len_sent)

#DIVIDE THE DEVIETION DATASET IN THE SUBSETS LIKE SENSEVAL2,3,7,...:
Xdev_s2 , Xdev_s3 , Xdev_7 , Xdev_13 , Xdev_15 =divide_dev(X_dev,datasets_dev)
Ydev_s2 , Ydev_s3 , Ydev_7 , Ydev_13 , Ydev_15 =divide_dev(Y_dev,datasets_dev)
ldev_s2 , ldev_s3 , ldev_7 , ldev_13 , ldev_15 =divide_dev(lentgh_of_sentences_dev,datasets_dev)


print("recupero dei dati concluso")

#NN HYPERPARAMETERS
print("number tags",ntags)
hidden_size= 400
NUM_STEPS=10#000
batch_size=20 #numero di sentences per batch
ntime_steps=50
lr=0.001

#NN CONSTRUCTION:
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('inputs'):
        # shape = (batch size, max length of sentence in batch)
        word_ids = tf.placeholder(tf.int32, shape=[None, None])
        # shape = (batch size)
        sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        # shape = (batch, sentence)
        labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        #ntime_steps =tf.reduce_max(sequence_lengths, reduction_indices=[0])

    with tf.name_scope("embedding_layer"):
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)
        # shape = (batch, sentence, word_vector_size)
        word_embeddings = tf.nn.embedding_lookup(L, word_ids)

    cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
    cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)

    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
        cell_bw, word_embeddings, sequence_length=sequence_lengths,
        dtype=tf.float32)

    H = tf.concat([output_fw, output_bw], axis=-1)

    #____________LSTM OUTPUT_________________________

    with tf.name_scope("LSTM_layer"):
        W = tf.get_variable("W", shape=[2*hidden_size, ntags],
                        dtype=tf.float32)

        b = tf.get_variable("b", shape=[ntags], dtype=tf.float32,
                        initializer=tf.zeros_initializer())

        context_rep_flat = tf.reshape(H, [-1, 2*hidden_size])
        pred = tf.matmul(context_rep_flat, W) + b
        lstm_Output = tf.reshape(pred, [-1, ntime_steps, ntags])


    #___________ATTENTIVE LAYER_____________________

    input_shape=H.get_shape().as_list()

    with tf.name_scope("Attentive_layer"):
        w= tf.Variable(tf.random_uniform(shape=(input_shape[-1],1),minval=-1,maxval=1))
        b=tf.zeros(shape=[ntime_steps])
        b=tf.Variable(b,validate_shape=False)

        u=tf.tensordot(H,w,axes=1)
        u=tf.squeeze(u,-1)
        u=u+b
        u=tf.tanh(u)
        a=tf.exp(u)

        mask = tf.sequence_mask(sequence_lengths,ntime_steps)
        if mask is not None:
            mask=tf.cast(mask,tf.float32)
            a=mask*a

        a/= tf.cast(tf.reduce_sum(a,axis=1,keepdims=True)+ tf.keras.backend.epsilon() , tf.float32) #K.epsilon()

        weighted_input=H*tf.expand_dims(a,-1)
        output=tf.reduce_sum(weighted_input,axis=1)

        att_output=tf.tile(output,[1,ntime_steps])
        att_output=tf.reshape(att_output,(-1,ntime_steps,2*hidden_size))

    #_________CONCATENATION LSTM OUTPUT E ATT LAYER________

    with tf.name_scope("Concatenation"):
        att_output= tf.concat([att_output,lstm_Output],-1)
        print("att_o",att_output.get_shape().as_list())

    #_________SOFTMAX LAYER________________________________

    softmax_scores= att_output
    print("s",softmax_scores.get_shape().as_list())
    with tf.name_scope("softmax_layer"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_scores, labels=labels)

    with tf.name_scope('mask_1'):
        mask = tf.sequence_mask(sequence_lengths)
        # apply mask
        losses = tf.boolean_mask(losses, mask)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)

    with tf.name_scope('prediction'):
        labels_pred = tf.cast(tf.argmax(softmax_scores, axis=-1), tf.int32)

    #THIS SECOND MASK IS BUILD taking into account that the words that I have interest in
    # classifying are those that have different X and Y :
    with tf.name_scope('mask_2'):
        mask1=tf.logical_not(tf.equal(word_ids,labels))
        l_p=tf.boolean_mask(labels_pred, mask1)
        l=tf.boolean_mask(labels, mask1)

    # accuracy calculated only on the words that I need to classify
    with tf.name_scope('WSD_accuracy'):
        correct_pred_WSD = tf.equal(l_p, l)
        accuracy_WSD = tf.reduce_mean(tf.cast(correct_pred_WSD, tf.float32))
        tf.summary.scalar('WSD_accuracy', accuracy_WSD)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(labels_pred, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()


display_step=100
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter("/home/lollo/Desktop/NLP/log", sess.graph)
    # Run the initializer
    sess.run(init)
    bar = tqdm.tqdm(range(NUM_STEPS))
    for step in bar:
        index_from=(step*batch_size)%NUM_SENTECES
        batch_x = [X[i%NUM_SENTECES] for i in range(index_from,index_from+batch_size)]
        len_sent=[lentgh_of_sentences[i%NUM_SENTECES] for i in range(index_from,index_from+batch_size)]

        batch_y = [Y[i%NUM_SENTECES] for i in range(index_from,index_from+batch_size)]

        # split the sentences too long
        batch_x, batch_y, len_sent = split_sentences(batch_x, batch_y, len_sent, ntime_steps)

        # pad all the sentences at the same length
        batch_x=create_padding(batch_x,ntime_steps)
        batch_y = create_padding(batch_y, ntime_steps)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={word_ids: batch_x,
                                      sequence_lengths:len_sent,
                                      labels: batch_y})
        # display the values and saves on tensorboard
        if step % display_step == 0 :
            summary_str = sess.run(summary, feed_dict={word_ids: batch_x,
                                      sequence_lengths:len_sent,
                                      labels: batch_y})
            writer.add_summary(summary_str, step)
            writer.flush()
            acc = sess.run(accuracy, feed_dict={word_ids: batch_x,
                                                #ntime_steps: max_len_sent,
                                                sequence_lengths:len_sent,
                                                labels: batch_y})
            acc_wsd = sess.run(accuracy_WSD, feed_dict={word_ids: batch_x,
                                                # ntime_steps: max_len_sent,
                                                sequence_lengths: len_sent,
                                                labels: batch_y})
            print("Step " + str(step)  + ", Training Accuracy= " + \
                  "{:.3f}".format(acc)+ ", Training Accuracy= " + \
                  "{:.3f}".format(acc_wsd))

    print("Optimization Finished!")
    print("Deviation set evaluation:")


    # FUNCTION THAT CALCULATE THE ACCURACY FOR AMBIGOUS WORDS PREDICTION IN DEV SETS
    def testWSD(Xd,Yd,ld):
        list_acc=[]
        #print(len(Xdev))
        bar = tqdm.tqdm(range(0,len(Xd),batch_size))
        for index in bar:
            batch_x = [Xd[i % len(Xd)] for i in range(index, index +batch_size)]
            batch_y = [Yd[i % len(Xd)] for i in range(index, index+batch_size)]
            len_sent = [ld[i % len(Xd)] for i in range(index, index+batch_size)]
            batch_x, batch_y, len_sent = split_sentences(batch_x, batch_y, len_sent, ntime_steps)
            max_len_sent = ntime_steps
            batch_x = create_padding(batch_x, max_len_sent)
            batch_y = create_padding(batch_y, max_len_sent)
            a=sess.run(accuracy_WSD, feed_dict={word_ids: batch_x,
                                                sequence_lengths:len_sent,
                                                labels: batch_y})
            print(a)
            if not math.isnan(a):
                list_acc.append(a)
        return sess.run(tf.reduce_mean(list_acc))
    flag=0
    if(flag==0):
        acc_S2=testWSD(Xdev_s2,Ydev_s2,ldev_s2)
        print("made 1/5")
        acc_S3 = testWSD(Xdev_s3, Ydev_s3, ldev_s3)
        print("made 2/5")
        acc_S7 = testWSD(Xdev_7, Ydev_7, ldev_7)
        print("made 3/5")
        acc_S13 = testWSD(Xdev_13, Ydev_13, ldev_13)
        print("made 4/5")
        acc_S15 = testWSD(Xdev_15, Ydev_15, ldev_15)
        print("made 5/5")
        print("batchsize: ", batch_size, " Num steps: ", NUM_STEPS, " hidden size: ", hidden_size, "max dimension sent: ",
          ntime_steps,"w2v size 300")
        print("Testing Accuracy s2:",acc_S2)
        print("Testing Accuracy s3:",acc_S3)
        print("Testing Accuracy s4:",acc_S7)
        print("Testing Accuracy s5:",acc_S13)
        print("Testing Accuracy s6:",acc_S15)
    elif flag==0:
        #_____________________TESTING PHASE___________________
        sys.exit("Error message")

        l, id, _, _ = build_testset('TEST/test_data.txt')
        X_test = build_vectDataset(l, final_voc)
        print(len(X_test))
        len_sent_test = [len(i) for i in X_test]
        batch_x_test, batch_y_test, len_sent = split_sentences(X_test, id, len_sent_test, sentence_dimension)
        max_len_sent = 50
        print(len_sent)

        batch_x = create_padding(batch_x_test, max_len_sent)
        print([len(i) for i in batch_x])
        #sys.exit("Error message")
        l_p = sess.run(labels_pred, feed_dict={word_ids: batch_x,
                                               sequence_lengths: len_sent})
        reversed_final_voc=dict((v,k) for k,v in final_voc.items())
        print(l_p[0])
        with open('results_test.txt', 'w') as f:
            for s,s1 in zip(id,l_p) :
                for b,b1 in zip(s,s1):
                    if b==0:
                        pass
                    else:
                        f.write("test."+b + '\t'+reversed_final_voc[b1]+"\n")


