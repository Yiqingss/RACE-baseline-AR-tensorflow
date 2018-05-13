import tensorflow as tf

def bilinear(hidden_size, h_q, h_doc, batch_size):
    with tf.variable_scope('bilinear'):  # Bilinear Layer (Attention Step)
        # bilinear attention
        W1 = tf.get_variable(shape=[hidden_size * 2, hidden_size * 2], dtype=tf.float32, name='W1',
                             initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        alpha = tf.matmul(h_q, W1)  # (bz, hz)
        alpha = tf.matmul(h_doc, tf.reshape(alpha, [batch_size, hidden_size * 2, 1]))
        alpha = tf.reshape(alpha, [batch_size, -1])  # (bz, len)
        alpha = tf.nn.softmax(alpha)  # (bz, len)
        attention = tf.multiply(tf.reshape(alpha, [batch_size, -1, 1]), h_doc)  # (bz, len, hz)
        att_d = tf.reduce_sum(attention, 1)  # (bz, hz)

        return att_d

def gate_attention(hidden_size, h_q, h_doc, batch_size):
    with tf.variable_scope('GA reader'):
        # doc: B x N x D
        # qry: B x Q x D
        # inter: B x N x Q
        # mask (qry): B x Q
        alphas_r = tf.nn.softmax(inter) * \
                   tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
        alphas_r = alphas_r / \
                   tf.expand_dims(tf.reduce_sum(alphas_r, axis=2), axis=-1)  # B x N x Q
        q_rep = tf.matmul(alphas_r, qry)  # B x N x D
        return eval(gating_fn)(doc, q_rep)

def pairwise_interaction(doc, qry):
    # doc: B x N x D
    # qry: B x Q x D
    shuffled = tf.transpose(qry, perm=[0, 2, 1])  # B x D x Q
    return tf.matmul(doc, shuffled)  # B x N x Q