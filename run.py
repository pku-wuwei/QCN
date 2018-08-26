#coding: utf-8
# v1.2 也可以先只取出每个句子，在batch_iter阶段转为id
# 这样好处是不需要一开始就将整个数据集进行转换，减小等待时间，减少内存消耗；
# 坏处是对每个句子每个epoch都要转换一遍，增加了整体的运行时间；

import tensorflow as tf
import numpy as np
import os 
import sys
import json 
import random
import time

from current_model import Config
from current_model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = str(3-Config.gpu_id)
vocab = {}

def get_word_and_char(sentence):
    words = sentence.split()
    get_id = lambda w : vocab[w] if w in vocab else 1
    word_ids = [get_id(w) for w in words]
    digitize_char = lambda x : ord(x) if 0<ord(x)<128 else 1            # 只有padding的才是0，非padding部分都大于0
    char_ids = [[digitize_char(c) for c in w] for w in words]
    return word_ids, char_ids

def digitize_data(fname):
    tasks = []
    start_time = time.time()
    with open('data/' + fname + '.txt') as fi:
        for line in fi:
            label, ps_string, pb_string, qt_string = line.strip().split('\t')
            ps_word, ps_char = get_word_and_char(ps_string)
            pb_word, pb_char = get_word_and_char(pb_string)
            qt_word, qt_char = get_word_and_char(qt_string)
            label = Config.label_list.index(label) if fname == 'train' else label
            tasks.append([ps_word, ps_char, pb_word, pb_char, qt_word, qt_char, label])
    print('loading {}:\tsize:{}\ttime:{}'.format(fname, len(tasks), time.time()-start_time))
    return tasks


# padding成固定shape,(b,m) (b,m,w),虽然各batch的b,m,w可能不同,
def batch_iter(data, batch_size, shuffle, is_train):
    num_batchs_per_epoch = int((len(data)-1)/batch_size) + 1
    num_epochs = 500 if is_train else 1
    for epoch in range(num_epochs):
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, len(data))
            batch_data = data[start_index:end_index]
            ps_words, ps_chars, pb_words, pb_chars, qt_words, qt_chars, label = list(zip(*batch_data))
            ps_words, ps_chars = pad_word_and_char(ps_words, ps_chars)
            pb_words, pb_chars = pad_word_and_char(pb_words, pb_chars)
            qt_words, qt_chars = pad_word_and_char(qt_words, qt_chars)
            yield [ps_words, pb_words, qt_words, ps_chars, pb_chars, qt_chars, np.array(label)]
            
def pad_word_and_char(words, chars):
    max_sent = min(len(max(words, key=len)), Config.max_sent)
    pad_sent = lambda x: x[:max_sent] if len(x)>max_sent else x+[0]*(max_sent-len(x))
    padded_sent = [pad_sent(sent) for sent in words]

    flatten_chars = [j for i in chars for j in i]
    max_word = max(min(len(max(flatten_chars, key=len)), Config.max_word), 5)
    pad_word = lambda x: x[:max_sent] if len(x)>max_sent else x+[[0]]*(max_sent-len(x))
    pad_char = lambda x: x[:max_word] if len(x)>max_word else x+[0]*(max_word-len(x))
    padded_word = [[pad_char(word) for word in pad_word(sent)] for sent in chars]
    return np.array(padded_sent), np.array(padded_word)

if __name__ == '__main__':

    embedding = []
    start_time = time.time()
    with open('data/embedding.txt') as fe:
        for i, line in enumerate(fe):
            items = line.split()
            vocab[items[0]]= i
            embedding.append(list(map(float,items[1:])))
    print('loading embedding:\twords:{}\ttime:{}'.format(len(embedding), time.time()-start_time))

    train_data = digitize_data('train')
    deva_data = digitize_data('dev15')
    devb_data = digitize_data('dev16')
    test_data = digitize_data('test17')
    print(vars(Config))

    model = Model(np.array(embedding))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config = tf_config) as sess:
        model.build_model()
        for v in tf.trainable_variables():
            print('name:{}\tshape:{}'.format(v.name,v.shape))
        print("ps_gating_pb p_concat_q")
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        if Config.restore and len([v for v in os.listdir('weights/') if '.index' in v]):
            saver.restore(sess, tf.train.latest_checkpoint('weights/'))
        batch_trains = batch_iter(train_data,Config.batch_size,True,True)
        losses = []
        show_time = time.time()
        total_steps = len(train_data)/Config.batch_size
        best_deva = best_devb = best_test = 0
        for step,batch_train in enumerate(batch_trains):
            batch_loss = model.train_batch(sess, batch_train)
            sys.stdout.write("\repoch:{:.4f}\t\t\tloss:{:.4f}".format(step/total_steps, batch_loss))
            losses.append(batch_loss)
            display_step = int(total_steps/2) if step<10*total_steps else int(total_steps/4)
            if step % display_step ==0:
                sys.stdout.write('\repoch:{:.4f}\t\taverage_loss:{:.4f}\n'.format(step/total_steps, sum(losses)/len(losses)))
                losses = []

                batch_devas = batch_iter(deva_data,Config.batch_size,False,False)
                with open('scorer/dev15/predict', 'w') as fw:
                    for batch_deva in batch_devas:
                        fw.write(model.test_batch(sess, batch_deva, True))
                result = os.popen('scorer/dev15/ev.pl scorer/dev15/gold scorer/dev15/predict').readlines()
                deva_print = result[-8].strip() + '\t' + result[-19].strip().split()[-1]
                best_deva = max(best_deva, float(deva_print.split()[8].rstrip('%')))

                batch_devbs = batch_iter(devb_data,Config.batch_size,False,False)
                with open('scorer/dev16/predict', 'w') as fw:
                    for batch_devb in batch_devbs:
                        fw.write(model.test_batch(sess, batch_devb))
                result = os.popen('python2 scorer/dev16/ev.py scorer/dev16/gold scorer/dev16/predict').readlines()
                devb_print = result[1].strip() + '\t' + result[8].strip() + '\t' + result[11].strip()
                best_devb = max(best_devb, float(devb_print.split()[6]))

                batch_tests = batch_iter(test_data,Config.batch_size,False,False)
                with open('scorer/test17/predict', 'w') as fw:
                    for batch_test in batch_tests:
                        fw.write(model.test_batch(sess, batch_test))
                result = os.popen('python2 scorer/test17/ev.py scorer/test17/gold scorer/test17/predict').readlines()
                test_print = result[1].strip() + '\t' + result[8].strip() + '\t' + result[11].strip()
                best_test = max(best_test, float(test_print.split()[6]))

                print('\ndeva:{}\ndevb:{}\ntest:{}\nshow_time:{:.4f}\tbest_deva:{}\t\tbest_devb:{}\t\tbest_test:{}\n'.format(deva_print, devb_print, test_print, time.time()-show_time, best_deva, best_devb, best_test))
                if (best_deva > 55) and (float(deva_print.split()[8].rstrip('%')) == best_deva):
                    saver.save(sess, 'weights/best', step)
                show_time = time.time()

