import tensorflow as  tf 
import numpy as np 

sess =tf.Session()
x=  tf.placeholder(dtype=tf.float16,shape=[None,50])
feed_dict =  {x:np.random.randint(5, size=(10, 50))}

k = tf.shape(x)[0]
top_span_range = tf.range(k) # [k]
k = tf.reshape(k , [1,])
z= tf.tile(top_span_range , k)
k = tf.shape(x)[0]
z= tf.reshape(z, [k,-1])


antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
antecedents_mask = tf.not_equal(antecedent_offsets, 0)   # [k, k]
# sess.run(antecedents_mask, feed_dict)

top_antecedent_emb = tf.gather(x, z) # [k, c, emb]
# o = sess.run(top_antecedent_emb, feed_dict)

target_emb = tf.expand_dims(x, 1) # [k, 1, emb]
target_emb = tf.tile(target_emb, [1, k, 1]) # [k, c, emb]
pair_emb = tf.concat([target_emb, top_antecedent_emb], 2) # [k, c, emb]

sess.run(top_antecedents_mask, feed_dict)