# inspect_model.py
import tensorflow as tf
import pickle
import h5py
import json

def inspect_and_fix_model():
    print("Inspecting model files...")
    
    # Check file existence
    import os
    if not os.path.exists('model_Adam.h5'):
        print("❌ model_Adam.h5 not found")
        return False
    
    if not os.path.exists('x_scaler.pkl'):
        print("❌ x_scaler.pkl not found")
        return False
    
    if not os.path.exists('y_scaler.pkl'):
        print("❌ y_scaler.pkl not found")
        return False
    
    # Inspect H5 file
    print("\nInspecting model_Adam.h5 structure:")
    with h5py.File('model_Adam.h5', 'r') as f:
        def print_attrs(name, obj):
            print(f"  {name}:")
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")
        
        f.visititems(print_attrs)
    
    # Try to load with different methods
    print("\nTrying to load model...")
    
    try:
        # Method 1: Try to load with custom_objects
        import keras
        print("Attempting to load with keras...")
        model = keras.models.load_model('model_Adam.h5')
        print("✅ Model loaded successfully with keras")
    except Exception as e1:
        print(f"❌ Failed with keras: {e1}")
        
        try:
            # Method 2: Try to load without custom objects
            print("\nAttempting to load without custom objects...")
            model = tf.keras.models.load_model('model_Adam.h5')
            print("✅ Model loaded successfully with tf.keras")
        except Exception as e2:
            print(f"❌ Failed with tf.keras: {e2}")
            
            try:
                # Method 3: Load model architecture and weights separately
                print("\nAttempting to load architecture and weights separately...")
                with h5py.File('model_Adam.h5', 'r') as f:
                    # Try to get model config
                    model_config = f.attrs.get('model_config', None)
                    if model_config:
                        if isinstance(model_config, bytes):
                            model_config = model_config.decode('utf-8')
                        model_config = json.loads(model_config)
                        print(f"Model config found: {model_config.keys()}")
                        
                        # Try to recreate model from config
                        model = tf.keras.models.model_from_json(model_config)
                        print("✅ Model created from config")
                        
                        # Try to load weights
                        model.load_weights('model_Adam.h5')
                        print("✅ Weights loaded successfully")
                    else:
                        print("❌ No model config found in file")
                        return False
            except Exception as e3:
                print(f"❌ Failed to load separately: {e3}")
                return False
    
    # Test scalers
    print("\nTesting scalers...")
    try:
        with open('x_scaler.pkl', 'rb') as f:
            x_scaler = pickle.load(f)
        print(f"✅ x_scaler loaded: {type(x_scaler)}")
        
        with open('y_scaler.pkl', 'rb') as f:
            y_scaler = pickle.load(f)
        print(f"✅ y_scaler loaded: {type(y_scaler)}")
        
        # Test a prediction
        print("\nTesting prediction...")
        test_input = [[0.1] * 10]  # Adjust based on your input dimension
        if hasattr(x_scaler, 'transform'):
            test_scaled = x_scaler.transform(test_input)
            prediction = model.predict(test_scaled, verbose=0)
            print(f"✅ Prediction successful: {prediction.shape}")
        else:
            print("⚠ x_scaler doesn't have transform method")
            
    except Exception as e:
        print(f"❌ Error with scalers: {e}")
        return False
    
    return True

if __name__ == "__main__":
    inspect_and_fix_model()