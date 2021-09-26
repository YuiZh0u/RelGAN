import tensorflow as tf
from utils.models.relational_memory import RelationalMemory
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from utils.ops import *


embedding_size = 16
filter_sizes = [3, 3, 3, 3, 3]
num_filters = [128, 256, 128, 256, 512]
num_blocks = [2, 2, 2, 2]

cnn_initializer = tf.compat.v1.keras.initializers.he_normal()
fc_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05)
# fc_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05) #If use Tensorflow v2


# The generator network based on the Relational Memory
def generator(x_real, temperature, vocab_size, batch_size, seq_len, gen_emb_dim, mem_slots, head_size, num_heads,
              hidden_dim, start_token):
    start_tokens = tf.constant([start_token] * batch_size, dtype=tf.int32)
    output_size = mem_slots * head_size * num_heads

    # build relation memory module
    g_embeddings = tf.compat.v1.get_variable('g_emb', shape=[vocab_size, gen_emb_dim],
                                   initializer=create_linear_initializer(vocab_size))
    gen_mem = RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
    g_output_unit = create_output_unit(output_size, vocab_size)

    # initial states
    init_states = gen_mem.initial_state(batch_size)

    # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
    gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)
    gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=seq_len, dynamic_size=False, infer_shape=True)
    gen_x_onehot_adv = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                                    infer_shape=True)  # generator output (relaxed of gen_x)

    # the generator recurrent module used for adversarial training
    def _gen_recurrence(i, x_t, h_tm1, gen_o, gen_x, gen_x_onehot_adv):
        mem_o_t, h_t = gen_mem(x_t, h_tm1)  # hidden_memory_tuple
        o_t = g_output_unit(mem_o_t)  # batch x vocab, logits not probs
        gumbel_t = add_gumbel(o_t)
        next_token = tf.stop_gradient(tf.argmax(input=gumbel_t, axis=1, output_type=tf.int32))
        next_token_onehot = tf.one_hot(next_token, vocab_size, 1.0, 0.0)

        x_onehot_appr = tf.nn.softmax(tf.multiply(gumbel_t, temperature))  # one-hot-like, [batch_size x vocab_size]

        # x_tp1 = tf.matmul(x_onehot_appr, g_embeddings)  # approximated embeddings, [batch_size x emb_dim]
        x_tp1 = tf.nn.embedding_lookup(params=g_embeddings, ids=next_token)  # embeddings, [batch_size x emb_dim]

        gen_o = gen_o.write(i, tf.reduce_sum(input_tensor=tf.multiply(next_token_onehot, x_onehot_appr), axis=1))  # [batch_size], prob
        gen_x = gen_x.write(i, next_token)  # indices, [batch_size]

        gen_x_onehot_adv = gen_x_onehot_adv.write(i, x_onehot_appr)

        return i + 1, x_tp1, h_t, gen_o, gen_x, gen_x_onehot_adv

    # build a graph for outputting sequential tokens
    _, _, _, gen_o, gen_x, gen_x_onehot_adv = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3, _4, _5: i < seq_len,
        body=_gen_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(params=g_embeddings, ids=start_tokens),
                   init_states, gen_o, gen_x, gen_x_onehot_adv))

    gen_o = tf.transpose(a=gen_o.stack(), perm=[1, 0])  # batch_size x seq_len
    gen_x = tf.transpose(a=gen_x.stack(), perm=[1, 0])  # batch_size x seq_len

    gen_x_onehot_adv = tf.transpose(a=gen_x_onehot_adv.stack(), perm=[1, 0, 2])  # batch_size x seq_len x vocab_size

    # ----------- pre-training for generator -----------------
    x_emb = tf.transpose(a=tf.nn.embedding_lookup(params=g_embeddings, ids=x_real), perm=[1, 0, 2])  # seq_len x batch_size x emb_dim
    g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)

    ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len)
    ta_emb_x = ta_emb_x.unstack(x_emb)

    # the generator recurrent moddule used for pre-training
    def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
        mem_o_t, h_t = gen_mem(x_t, h_tm1)
        o_t = g_output_unit(mem_o_t)
        g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch_size x vocab_size
        x_tp1 = ta_emb_x.read(i)
        return i + 1, x_tp1, h_t, g_predictions

    # build a graph for outputting sequential tokens
    _, _, _, g_predictions = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3: i < seq_len,
        body=_pretrain_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(params=g_embeddings, ids=start_tokens),
                   init_states, g_predictions))

    g_predictions = tf.transpose(a=g_predictions.stack(),
                                 perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

    # pre-training loss
    pretrain_loss = -tf.reduce_sum(
        input_tensor=tf.one_hot(tf.cast(tf.reshape(x_real, [-1]), dtype=tf.int32), vocab_size, 1.0, 0.0) * tf.math.log(
            tf.clip_by_value(tf.reshape(g_predictions, [-1, vocab_size]), 1e-20, 1.0)
        )
    ) / (seq_len * batch_size)

    return gen_x_onehot_adv, gen_x, pretrain_loss, gen_o


def discriminator(x_onehot, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn, is_train=True):

    # ============= Embedding Layer =============
    d_embeddings = tf.compat.v1.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
                                   initializer=create_linear_initializer(vocab_size))
    input_x_re = tf.reshape(x_onehot, [-1, vocab_size])
    emb_x_re = tf.matmul(input_x_re, d_embeddings)
    emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim])  # batch_size x seq_len x dis_emb_dim
    emb_x_expanded = tf.expand_dims(emb_x, 2)  # batch_size x seq_len x 1 x emd_dim

    # ============= First Convolution Layer =============
    with tf.compat.v1.variable_scope("conv-0"):
        conv0 = tf.compat.v1.layers.conv2d(
            emb_x_expanded,
            filters=num_filters[0],
            kernel_size=[filter_sizes[0], 1],
            kernel_initializer=cnn_initializer,
            activation=tf.nn.relu)

    # ============= Convolution Blocks =============
    conv1 = conv_block(conv0, 1, max_pool=True, is_train=is_train)

    conv2 = conv_block(conv1, 2, max_pool=False, is_train=is_train)

    # conv3 = conv_block(conv2, 3, max_pool=False, is_train=is_train)
    #
    # conv4 = conv_block(conv3, 4, max_pool=False, is_train=is_train)

    # ============= k-max Pooling =============
    h = tf.transpose(a=tf.squeeze(conv2), perm=[0, 2, 1])
    top_k = tf.nn.top_k(h, k=1, sorted=False).values
    h_flat = tf.reshape(top_k, [batch_size, -1])

    # ============= Fully Connected Layers =============
    # fc1_out = tf.layers.dense(h_flat, 2048, activation=tf.nn.relu, kernel_initializer=fc_initializer)
    #
    # fc2_out = tf.layers.dense(fc1_out, 2048, activation=tf.nn.relu, kernel_initializer=fc_initializer)

    logits = tf.compat.v1.layers.dense(h_flat, 1, activation=None, kernel_initializer=fc_initializer)

    logits = tf.squeeze(logits, -1)  # batch_size

    return logits


def conv_block(input, i, max_pool=True, is_train=True):
    with tf.compat.v1.variable_scope("conv-block-%s" % i):
        # Two "conv-batch_norm-relu" layers.
        for j in range(2):
            with tf.compat.v1.variable_scope("conv-%s" % j):
                # convolution
                conv = tf.compat.v1.layers.conv2d(
                    input,
                    filters=num_filters[i],
                    kernel_size=[filter_sizes[i], 1],
                    kernel_initializer=cnn_initializer,
                    activation=None)
                # batch normalization
                conv = tf.compat.v1.layers.batch_normalization(conv, training=is_train)
                # relu
                conv = tf.nn.relu(conv)

        if max_pool:
            # Max pooling
            pool = tf.compat.v1.layers.max_pooling2d(
                conv,
                pool_size=(3, 1),
                strides=(2, 1),
                padding="SAME")
            return pool
        else:
            return conv