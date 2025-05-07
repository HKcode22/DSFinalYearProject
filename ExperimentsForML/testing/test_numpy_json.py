import numpy as np
import json

def convert_numpy_to_serializable(obj):
    """Convert NumPy objects to JSON serializable Python types"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.datetime64):
        return str(obj)
    elif isinstance(obj, np.complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_serializable(item) for item in obj]
    return obj

# Create test data with various NumPy types
test_data = {
    'array': np.array([1, 2, 3]),
    'int': np.int64(10),
    'float': np.float32(3.14),
    'bool': np.bool_(True),
    'datetime': np.datetime64('2023-01-01'),
    'complex': np.complex128(1+2j),
    'nested': {
        'array': np.array([4, 5, 6])
    }
}

# Convert and serialize to JSON
serializable_data = convert_numpy_to_serializable(test_data)
json_str = json.dumps(serializable_data, indent=2)

print("Test data converted successfully to JSON:")
print(json_str)

# Test the resulting JSON
parsed = json.loads(json_str)
print("\nVerification:")
print(f"Array converted: {isinstance(parsed['array'], list)}")
print(f"Integer converted: {isinstance(parsed['int'], int)}")
print(f"Float converted: {isinstance(parsed['float'], float)}")
print(f"Bool converted: {isinstance(parsed['bool'], bool)}")
print(f"Datetime converted: {isinstance(parsed['datetime'], str)}")
print(f"Complex converted: {isinstance(parsed['complex'], dict)}")
print(f"Nested array converted: {isinstance(parsed['nested']['array'], list)}") 