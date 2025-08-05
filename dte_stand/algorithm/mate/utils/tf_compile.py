import functools
import tensorflow as tf

def maybe_compile(flag_name):
    ''' 
    Decorator that wraps class instance method in a tf.function if the flag is set
    Allows performance improvements for code where only some execution paths can 
    be compiled with tf.function.
    '''
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if getattr(self, flag_name, False):
                cache_attr = f"_cached_tf_function_{method.__name__}"
                if not hasattr(self, cache_attr):
                    setattr(self, cache_attr, tf.function(method))
                tf_method = getattr(self, cache_attr)
                return tf_method(self, *args, **kwargs)
            else:
                return method(self, *args, **kwargs)
        return wrapper
    
    return decorator