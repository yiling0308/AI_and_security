##### 執行chapter1資料夾中的first_session_only_tensorflow.py檔

    python2 first_session_only_tensorflow.py
 
### 出現錯誤
#### 警告 initialize_all_variables 即將在2017-03-02過期，且將被刪除，請改為使用tf.global_variables_initializer 
```
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py:170: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
2018-10-03 09:58:30.444616: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 09:58:30.444641: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 09:58:30.444647: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 09:58:30.444650: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
10
```
##### 修改後程式碼
```
import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9,name='y')


model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
```
##### 修改後答案
```
2018-10-03 10:23:51.462025: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 10:23:51.462053: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 10:23:51.462057: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 10:23:51.462061: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
10
```
##### 執行chapter1資料夾中的first_session.py檔

    python2 first_session.py
    
##### 答案    
```
10
<tf.Variable 'y:0' shape=() dtype=int32_ref>
```
##### 執行chapter1資料夾中的using_tensorboard.py檔

    python2 using_tensorboard.py 
    
##### 答案    
```
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py:170: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
2018-10-03 10:31:16.454821: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 10:31:16.454839: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 10:31:16.454846: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-10-03 10:31:16.454851: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
Traceback (most recent call last):
  File "using_tensorboard.py", line 11, in <module>
    merged = tf.merge_all_summaries()
AttributeError: 'module' object has no attribute 'merge_all_summaries'
ERROR:tensorflow:==================================
Object was never used (type <class 'tensorflow.python.framework.ops.Operation'>):
<tf.Operation 'init' type=NoOp>
If you want to mark it as used call its "mark_used()" method.
It was originally created here:
['File "using_tensorboard.py", line 8, in <module>\n    model = tf.initialize_all_variables()', 'File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py", line 170, in wrapped\n    return _add_should_use_warning(fn(*args, **kwargs))', 'File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py", line 139, in _add_should_use_warning\n    wrapped = TFShouldUseWarningWrapper(x)', 'File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py", line 96, in __init__\n    stack = [s.strip() for s in traceback.format_stack()]']
==================================
```
##### 執行chapter1資料夾中的programming_model.py檔

    python2 programming_model.py
    
##### 答案    
```
Traceback (most recent call last):
  File "programming_model.py", line 6, in <module>
    y = tf.mul(a,b)
AttributeError: 'module' object has no attribute 'mul'
```
