import numpy as np
import json

def convert_numpy_to_serializable(obj):
    """
    Convert NumPy objects to JSON serializable Python types.
    
    This function recursively converts NumPy arrays, scalars, and other
    non-serializable types into their Python equivalents.
    
    Args:
        obj: The object to convert, can be a NumPy array, scalar, or a nested structure
            containing NumPy objects.
            
    Returns:
        A JSON-serializable version of the input object.
    """
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

def json_dump_numpy(obj, fp, indent=2, **kwargs):
    """
    Dump an object with NumPy arrays/values to a JSON file.
    
    Args:
        obj: The object to serialize
        fp: File-like object to write to
        indent: Indentation level for pretty-printing
        **kwargs: Additional arguments passed to json.dump
    """
    json.dump(convert_numpy_to_serializable(obj), fp, indent=indent, **kwargs)

def json_dumps_numpy(obj, indent=2, **kwargs):
    """
    Convert an object with NumPy arrays/values to a JSON string.
    
    Args:
        obj: The object to serialize
        indent: Indentation level for pretty-printing
        **kwargs: Additional arguments passed to json.dumps
        
    Returns:
        str: JSON string representation
    """
    return json.dumps(convert_numpy_to_serializable(obj), indent=indent, **kwargs) 