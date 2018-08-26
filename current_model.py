# coding: utf-8
import tensorflow as tf
import numpy as np
from utils import ps_pb_interaction, pq_interaction, source2token, dense, multilayer_highway


class Config(object):
    word2vec_init   = True
    word2vec_size   = 300
    hidden_size = 300
    batch_size = 8
    learning_rate = 0.05
    l2_weight = 1e-6
    dropout = 1.0
    max_sent = 150
    max_word = 20
    label_list = ['Good','PotentiallyUseful','Bad']
    write_list = ['true', 'false', 'false']
    restore = False
    gpu_id = 0

    char_emb = 100
    filter_sizes = [2,3,4,5]
    filter_num = 25


class Model(object):
    def __init__(self, embedding):
        self.word_embedding = embedding

    def build_model(self):
        with tf.variable_scope("attention_model",initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.ps_words = tf.placeholder(tf.int32, [None, None])                       # (b,m)
            self.pb_words = tf.placeholder(tf.int32, [None, None])                       
            self.qt_words = tf.placeholder(tf.int32, [None, None])                    

            self.ps_length= tf.reduce_sum(tf.sign(self.ps_words),1)                       # (b,)
            self.pb_length= tf.reduce_sum(tf.sign(self.pb_words),1)                       
            self.qt_length= tf.reduce_sum(tf.sign(self.qt_words),1)                       # (b,m,1)
            self.ps_mask = tf.expand_dims(tf.sequence_mask(self.ps_length, tf.shape(self.ps_words)[1], tf.float32), -1)
            self.pb_mask = tf.expand_dims(tf.sequence_mask(self.pb_length, tf.shape(self.pb_words)[1], tf.float32), -1)
            self.qt_mask = tf.expand_dims(tf.sequence_mask(self.qt_length, tf.shape(self.qt_words)[1], tf.float32), -1)

            self.ps_chars = tf.placeholder(tf.int32, [None, None, None])                 # (b,m,w)
            self.pb_chars = tf.placeholder(tf.int32, [None, None, None])                 
            self.qt_chars = tf.placeholder(tf.int32, [None, None, None])                 

            self.ps_fchar = tf.reshape(self.ps_chars, [-1, tf.shape(self.ps_chars)[2]])    # (bm,w)
            self.pb_fchar = tf.reshape(self.pb_chars, [-1, tf.shape(self.pb_chars)[2]])
            self.qt_fchar = tf.reshape(self.qt_chars, [-1, tf.shape(self.qt_chars)[2]])
            self.ps_clength= tf.reduce_sum(tf.sign(self.ps_fchar),1)                      # (bm,)
            self.pb_clength= tf.reduce_sum(tf.sign(self.pb_fchar),1)                      
            self.qt_clength= tf.reduce_sum(tf.sign(self.qt_fchar),1)                      # (bm,w,1)
            self.ps_cmask = tf.expand_dims(tf.sequence_mask(self.ps_clength, tf.shape(self.ps_fchar)[1], tf.float32), -1)
            self.pb_cmask = tf.expand_dims(tf.sequence_mask(self.pb_clength, tf.shape(self.pb_fchar)[1], tf.float32), -1)
            self.qt_cmask = tf.expand_dims(tf.sequence_mask(self.qt_clength, tf.shape(self.qt_fchar)[1], tf.float32), -1)

            self.is_train = tf.placeholder(tf.bool)
            self.dropout = tf.cond(self.is_train, lambda: Config.dropout, lambda: 1.0)
            self.labels = tf.placeholder(tf.int32, [None])                          # (b,)

            with tf.device('/cpu:0'):
                self.embed_matrix = tf.convert_to_tensor(self.word_embedding, dtype=tf.float32)
                self.ps_emb = tf.nn.embedding_lookup(self.embed_matrix, self.ps_words)        # (b,m,d)
                self.pb_emb = tf.nn.embedding_lookup(self.embed_matrix, self.pb_words)      
                self.qt_emb = tf.nn.embedding_lookup(self.embed_matrix, self.qt_words)      

                #self.char_embedding = tf.get_variable('char', [128, Config.char_emb])
                #self.ps_cemb = tf.nn.embedding_lookup(self.char_embedding, self.ps_fchar)     # (bm,w,d)->(bm,d)->(b,m,d)
                #self.pb_cemb = tf.nn.embedding_lookup(self.char_embedding, self.pb_fchar)     
                #self.qt_cemb = tf.nn.embedding_lookup(self.char_embedding, self.qt_fchar)     

            #with tf.variable_scope("char") as scope:
                #ps_cinp = source2token(self.ps_cemb, self.ps_cmask, self.dropout, 'char')
                #scope.reuse_variables()
                #pb_cinp = source2token(self.pb_cemb, self.pb_cmask, self.dropout, 'char')
                #qt_cinp = source2token(self.qt_cemb, self.qt_cmask, self.dropout, 'char')

            #with tf.variable_scope("input") as scope:
                #self.ps_inp = tf.concat([self.ps_emb, tf.reshape(ps_cinp, [tf.shape(self.ps_chars)[0],-1,Config.char_emb])], -1)
                #self.pb_inp = tf.concat([self.pb_emb, tf.reshape(pb_cinp, [tf.shape(self.pb_chars)[0],-1,Config.char_emb])], -1)
                #self.qt_inp = tf.concat([self.qt_emb, tf.reshape(qt_cinp, [tf.shape(self.qt_chars)[0],-1,Config.char_emb])], -1)
                #self.ps_input = multilayer_highway(self.ps_emb, Config.hidden_size, 1, tf.nn.elu, self.dropout, 'ps_input')
                #self.pb_input = multilayer_highway(self.pb_emb, Config.hidden_size, 1, tf.nn.elu, self.dropout, 'pb_input')
                #self.qt_input = dense(self.qt_emb, Config.word2vec_size, tf.nn.tanh, self.dropout, 'qt_input')


            with tf.variable_scope("pq_interaction"):
                para = ps_pb_interaction(self.ps_emb, self.pb_emb, self.ps_mask, self.pb_mask, self.dropout, 'parallel')
                orth = ps_pb_interaction(self.ps_emb, self.pb_emb, self.ps_mask, self.pb_mask, self.dropout, 'orthogonal')
                self.p = tf.concat([para, orth], -1)
                self.q = tf.layers.dense(self.qt_emb,2*Config.hidden_size,tf.nn.tanh,name='qt_tanh') * tf.layers.dense(self.qt_emb,2*Config.hidden_size,tf.nn.sigmoid,name='qt_sigmoid')
                p_inter, q_inter = pq_interaction(self.p, self.q, self.ps_mask, self.qt_mask, self.dropout, 'p_q')
                self.p_vec = source2token(p_inter, self.ps_mask, self.dropout, 'p_vec')
                self.q_vec = source2token(q_inter, self.qt_mask, self.dropout, 'q_vec')

            with tf.variable_scope("loss"):
                l0 = tf.concat([self.p_vec, self.q_vec], 1)
                l1 = tf.layers.dense(l0, 300, tf.nn.elu, name='l1')
                l2 = tf.layers.dense(l1, 300, tf.nn.elu, name='l2')
                self.logits = tf.layers.dense(l2, 3, tf.identity, name='logits')
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels,3,dtype=tf.float32), logits=self.logits),-1)
                for v in tf.trainable_variables():
                    self.loss += Config.l2_weight * tf.nn.l2_loss(v)
                self.train_op = tf.train.GradientDescentOptimizer(Config.learning_rate).minimize(self.loss)

    def train_batch(self, sess, batch_data):
        ps_words, pb_words, qt_words, ps_chars, pb_chars, qt_chars, labels = batch_data
        feed = {self.ps_words: ps_words,
                self.pb_words: pb_words,
                self.qt_words: qt_words,
                self.ps_chars: ps_chars,
                self.pb_chars: pb_chars,
                self.qt_chars: qt_chars,
                self.labels: labels,
                self.is_train: True
               }
        _, loss = sess.run([self.train_op, self.loss], feed_dict = feed)
        return loss

    def test_batch(self, sess, batch_test, is_deva=False):
        ps_words, pb_words, qt_words, ps_chars, pb_chars, qt_chars, cids = batch_test
        feed = {self.ps_words: ps_words,
                self.pb_words: pb_words,
                self.qt_words: qt_words,
                self.ps_chars: ps_chars,
                self.pb_chars: pb_chars,
                self.qt_chars: qt_chars,
                self.is_train: False
               }
        logits = sess.run(self.logits, feed_dict = feed)
        score = logits[:,0]
        predict = np.argmax(logits,1)
        return_string = ''
        for i in range(len(cids)):
            if is_deva:
                wstring = cids[i]+'\t'+ Config.label_list[predict[i]][:9]+'\n'
            else:
                wstring = '_'.join(cids[i].split('_')[:2])+'\t'+cids[i]+'\t0\t'+str(score[i])+'\t'+Config.write_list[predict[i]]+'\n'
            return_string += wstring
        return return_string
