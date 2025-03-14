import tensorflow as tf

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.config.list_physical_devices('GPU'))

# Ensure that TensorFlow is using the GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set TensorFlow to use memory growth on GPU for efficiency
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    except RuntimeError as e:
        # Memory growth must be set before initializing GPUs
        print(str(e))
else:
    print("GPU is not available, using CPU instead.")

# Perform a simple tensor operation
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
c = tf.matmul(a, b)

print("Result of matrix multiplication:")
print(c.numpy())
