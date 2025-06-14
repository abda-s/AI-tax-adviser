import tensorflow as tf
import numpy as np

def test_tensorflow():
    print("Testing TensorFlow installation...")
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Test basic TensorFlow operations
    print("\nTesting basic operations...")
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:")
    print(c.numpy())
    
    # Test model creation
    print("\nTesting model creation...")
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        print("Successfully created a simple model")
        
        # Test model prediction
        test_input = np.random.random((1, 5))
        prediction = model.predict(test_input)
        print("Successfully made a prediction")
        print(f"Prediction shape: {prediction.shape}")
        
    except Exception as e:
        print(f"Error during model testing: {str(e)}")
    
    print("\nTensorFlow test completed")

if __name__ == '__main__':
    test_tensorflow() 