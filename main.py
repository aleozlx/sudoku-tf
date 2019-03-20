import numpy as np
import tensorflow as tf

x = tf.Variable(np.array([
    0, 0, 7, 0, 0, 0, 3, 0, 2,
    2, 0, 0, 0, 0, 5, 0, 1, 0,
    0, 0, 0, 8, 0, 1, 4, 0, 0,
    0, 1, 0, 0, 9, 6, 0, 0, 8,
    7, 6, 0, 0, 0, 0, 0, 4, 9,
    0, 0, 0, 0, 0, 0 ,0, 0, 0,
    0, 0, 0, 1, 0, 3, 0, 0, 0,
    8, 0, 1, 0, 6, 0, 0, 0, 0,
    0, 0, 0 ,7, 0, 0, 0 ,6, 3
]).reshape(9,9))

dom_zero = tf.equal(x, 0)
n = tf.reduce_sum(tf.cast(dom_zero, tf.int32))
dom0 = tf.broadcast_to(dom_zero, (9,9,9))
domH = tf.stack([
    tf.tile(
        tf.reshape(
            tf.logical_not(tf.reduce_any(tf.equal(x, v), 1)),
        (9, 1)),
    (1, 9)) for v in range(1,10)
])
domV = tf.stack([
    tf.reshape(
        tf.tile(
            tf.logical_not(tf.reduce_any(tf.equal(x, v), 0)),
        (9,)),
    (9,9)) for v in range(1,10)
])

domB = tf.reshape(tf.transpose(
    tf.reshape(tf.tile(
        tf.equal(
            tf.transpose(tf.nn.max_pool(
                tf.reshape(
                    tf.stack([
                        tf.cast(tf.equal(x, v), tf.int32)
                            for v in range(1,10)
                    ], axis=2),
                (1,9,9,9)),
                (1,3,3,1),
                (1,3,3,1),
                'VALID',
            ), (3,1,2,0)),
        0), # shape: 9,3,3,1
    (1,1,1,9)), (9,3,3,3,3)),
(0,1,3,2,4)), (9,9,9))
dom = tf.cast(tf.logical_and(
    dom0, tf.logical_and(
        domB, tf.logical_and(
            domH, domV
        )
    )
), tf.int32)

with tf.device('cpu:0'):
    ct0 = tf.where(tf.equal(tf.reduce_sum(dom, 0), 1))
    ctV = tf.where(tf.equal(tf.reduce_sum(dom, 1), 1))
    ctH = tf.where(tf.equal(tf.reduce_sum(dom, 2), 1))

    blks = [
        tf.constant(np.array([ [0,0], [0,1], [0,2],
                               [1,0], [1,1], [1,2],
                               [2,0], [2,1], [2,2] ]) + blk_ptr)
            for blk_ptr in [[0,0], [0,3], [0,6], [3,0], [3,3], [3,6], [6,0], [6,3], [6,6]]
    ]
    blk_cubes = [ # block*(in-block, vals) = 9*(9,9)
        tf.gather_nd(tf.transpose(dom, (1,2,0)), blk) for blk in blks
    ]
    ctB = [ # values per block: block*(#vals, ) = 9*?
        tf.reshape(tf.where(tf.equal(tf.reduce_sum(blk_cube, axis=0), 1)), [-1])
            for blk_cube in blk_cubes
    ]

with tf.device('cpu:0'):
    g0 = tf.argmax(tf.gather_nd(tf.transpose(dom, (1,2,0)), ct0), axis=1)
    gV = tf.argmax(tf.gather_nd(tf.transpose(dom, (0,2,1)), ctV), axis=1)
    gH = tf.argmax(tf.gather_nd(dom, ctH), axis=1)
    gB = [ # in-block indices: block*(in-block, )
        tf.argmax(tf.gather(tf.transpose(blk_cube), ctb), axis=1)
            for ctb, blk_cube in zip(ctB, blk_cubes)
    ]
    locV = tf.stack([gV, ctV[:, 1]], axis=1)
    locH = tf.stack([ctH[:, 1], gH], axis=1)
    locB = tf.concat([ # coordinate transform
        tf.gather(blk, gb) for gb, blk in zip(gB, blks)
    ], axis=0)

    valV = ctV[:, 0]
    valH = ctH[:, 0]
    valB = tf.concat(ctB, axis=0) # update values

    idxUpdates = tf.concat([ct0, locV, locH, locB], axis=0)
    valUpdates = tf.concat([g0, valV, valH, valB], axis=0) + 1
    reasons = tf.concat([
        tf.fill(tf.shape(g0), tf.constant("constraint")),
        tf.fill(tf.shape(valV), tf.constant("v-scan")),
        tf.fill(tf.shape(valH), tf.constant("h-scan"), ),
        tf.fill(tf.shape(valB), tf.constant("blk-scan"))
    ], axis=0)

    update_op = tf.scatter_nd_update(x, idxUpdates, valUpdates)

init = tf.global_variables_initializer()

from tqdm import tqdm
config = tf.ConfigProto(
    device_count={'GPU':1},
    allow_soft_placement=True,
    log_device_placement=True,
    inter_op_parallelism_threads=2,
    intra_op_parallelism_threads=2
)

with tf.Session(config=config) as sess:
#with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=2)) as sess:
    for _ in tqdm(range(10000)):
        sess.run(init)
        for step in range(sess.run(n)):
            idx, val,  _ = sess.run([
                idxUpdates, valUpdates, update_op])
            if len(val) == 0:
                break
        #print(sess.run(x))


