# original paper setting: -/77.21/87.85
# add char-cnn 200d: 57.30/77.64/87.75
# 1 layer highway body->subject<->comment: 56/76/86
# 1 layer unshared tanh body->subject<->comment: 57.51/77.18/86.41
import tensorflow as tf


# inpa,inpb (b,m,d) maska,maskb (b,m,1)
def ps_pb_interaction(ps, pb, ps_mask, pb_mask, keep_prob, scope):
    with tf.variable_scope(scope):
        b, m, n, d = tf.shape(ps)[0], tf.shape(ps)[1], tf.shape(pb)[1], ps.get_shape().as_list()[2]
        attn_mask = tf.expand_dims(ps_mask*tf.reshape(pb_mask,[b,1,n]), -1)             # (b,m,n,1)

        head = tf.tile(tf.expand_dims(ps, 2), [1,1,n,1])                          # (b,m,1,d)
        tail = tf.tile(tf.expand_dims(pb, 1), [1,m,1,1])                          # (b,1,n,d)
        parallel = head*tf.reduce_sum(head*tail, -1, True)/(tf.reduce_sum(head*head, -1, True)+1e-5)
        orthogonal = tail - parallel
        base = parallel if scope == 'parallel' else orthogonal

        interaction = dense(base, d, scope='interaction')    # (b,m,n,d)
        logits = 10.0*tf.tanh((interaction)/10.0) + (1 - attn_mask) * (-1e30)

        attn_score = tf.nn.softmax(logits, 2) * attn_mask
        attn_result = tf.reduce_sum(attn_score * tail, 2)                           # (b,m,d)
        fusion_gate = dense(tf.concat([ps, attn_result], -1), d, tf.sigmoid, scope='fusion_gate')*ps_mask
        return (fusion_gate*ps + (1-fusion_gate)*attn_result) * ps_mask

# inpa,inpb (b,m,d) maska,maskb (b,m,1)
def pq_interaction(ps, qt, ps_mask, qt_mask, keep_prob, scope):
    with tf.variable_scope(scope):
        b, m, n, d = tf.shape(ps)[0], tf.shape(ps)[1], tf.shape(qt)[1], ps.get_shape().as_list()[2]
        attn_mask = tf.expand_dims(ps_mask*tf.reshape(qt_mask,[b,1,n]), -1)             # (b,m,n,1)

        head = tf.tile(tf.expand_dims(ps, 2), [1,1,n,1])                          # (b,m,1,d)
        tail = tf.tile(tf.expand_dims(qt, 1), [1,m,1,1])                          # (b,1,n,d)
        interaction = dense(tf.concat([head, tail], -1), d, scope='interaction')    # (b,m,n,d)
        #interaction = tf.reduce_sum(head*tail, -1, True)
        #interaction = tf.reduce_sum(dense(head, d, scope='interaction')*tail, -1, True)
        logits = 5.0*tf.tanh((interaction)/5.0) + (1 - attn_mask) * (-1e30)

        atta_score = tf.nn.softmax(logits, 2) * attn_mask
        atta_result = tf.reduce_sum(atta_score * tail, 2)                           # (b,m,d)

        attb_score = tf.nn.softmax(logits, 1) * attn_mask
        attb_result = tf.reduce_sum(attb_score * head, 1)                       # (b,n,d)

        cata = tf.concat([ps, atta_result], -1) * ps_mask
        catb = tf.concat([qt, attb_result], -1) * qt_mask
        suba = (ps - atta_result) * (ps - atta_result) * ps_mask
        subb = (qt - attb_result) * (qt - attb_result) * qt_mask
        mula = ps * atta_result * ps_mask
        mulb = qt * attb_result * qt_mask
        nna = dense(tf.concat([suba, mula], -1), d, tf.nn.elu, scope='nna') * ps_mask
        nnb = dense(tf.concat([subb, mulb], -1), d, tf.nn.elu, scope='nnb') * qt_mask
        #return tf.concat([suba, mula], -1)*ps_mask, tf.concat([subb, mulb], -1)*qt_mask
        return cata, catb

def source2token(rep_tensor, rep_mask, keep_prob, scope):
    with tf.variable_scope(scope):
        ivec = rep_tensor.get_shape().as_list()[2]
        map1 = dense(rep_tensor, ivec, tf.nn.elu, keep_prob, 'map1')*rep_mask  # (b,n,d)
        map2 = dense(rep_tensor, ivec, tf.identity, keep_prob, 'map2')*rep_mask  # (b,n,d)
        map2_masked = map1 + (1-rep_mask) * (-1e30)
        soft = tf.nn.softmax(map2_masked, 1)*rep_mask   # bs,sl,vec
        return tf.reduce_sum(soft * rep_tensor, 1)      # bs, vec

def dense(inp, out_size, activation=tf.identity, keep_prob=1.0, scope=None, need_bias=True):
    with tf.variable_scope(scope):
        inp_shape = [inp.get_shape().as_list()[i] or tf.shape(inp)[i] for i in range(len(inp.get_shape().as_list()))]
        input = tf.nn.dropout(tf.reshape(inp, [-1, inp_shape[-1]]), keep_prob)
        W = tf.get_variable('W', shape=[input.get_shape()[-1],out_size],dtype=tf.float32)
        b = tf.get_variable('b', shape=[out_size], dtype=tf.float32, initializer=tf.zeros_initializer()) if need_bias else 0
        return activation(tf.reshape(tf.matmul(input, W) + b, inp_shape[:-1] + [out_size]))

def multilayer_highway(inp, out_size, layers, activation=tf.nn.elu, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope):
        output = inp
        for i in range(layers):
            line = dense(output, out_size, tf.identity, keep_prob, 'line'+str(i))
            gate = dense(output, out_size, tf.sigmoid, keep_prob, 'gate'+str(i))
            tran = dense(output, out_size, activation, keep_prob, 'tran'+str(i))
            output = gate*tran + (1-gate)*line
        return output

def text_cnn(input, filter_sizes, filter_num):
    input_expand = tf.expand_dims(input, -1)    # (b,m,d,1)
    embedding_size = tf.shape(input)[2]
    output_list = []
    for filter_size in filter_sizes:
        with tf.variable_scope("conv{}".format(filter_size)):
            filter_shape = tf.convert_to_tensor([filter_size, input.get_shape().as_list()[2], 1, filter_num])
            filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')                # tensorflow 需要trainable variables 拥有确定的shape
            conv = tf.nn.conv2d(input_expand, filter, [1,1,1,1], 'VALID', name='conv')
            hidden = tf.nn.elu(conv+ tf.Variable(tf.constant(0.1, shape=[filter_num])))
            pooling = tf.reduce_max(hidden, 1, True)    # tf.nn.max_pool 需要ksize是int
            output_list.append(tf.squeeze(pooling, [1,2]))
    return tf.concat(output_list,1)

def text_gru(input, input_length, hidden_size, keep_prob, mode='states'):
    fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size), keep_prob, keep_prob)
    bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size), keep_prob, keep_prob)
    outs, states=tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input, input_length, dtype=tf.float32)
    if mode == 'states':
        return tf.concat(states, 1)
    elif mode == 'max':
        return tf.reduce_max(tf.concat(outs, 2), 1)
    elif mode == 'mean':
        return tf.reduce_mean(tf.concat(outs,2), 1)

    

