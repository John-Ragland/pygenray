"""
MATLAB Wrapper for PyGenRay

This module provides MATLAB-compatible functions for ray tracing calculations.
Key functions are designed to handle MATLAB's array passing conventions robustly.

To reload this module from MATLAB after making changes:
    py.importlib.reload(py.importlib.import_module('pygenray.matlab_wrapper'))

Example MATLAB usage:
    [xray, tray, zray, pray, dzpdz] = py.pygenray.matlab_wrapper.rayintrd_bathy(...
        y0s, dx, r, c, cp, cpp, z, xin, D, xbot, bot, bot_slope, pyargs('debug', true));
"""

import pygenray as pr
import numpy as np
import xarray as xr
import array


def reload_module():
    """
    Helper function to reload this module from MATLAB.
    Call this from MATLAB as: py.pygenray.matlab_wrapper.reload_module()
    """
    import importlib
    import sys
    module_name = 'pygenray.matlab_wrapper'
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        print(f"Reloaded {module_name}")
    else:
        print(f"Module {module_name} not found in sys.modules")

def rayintrd_bathy(
        y0s,
        dx,
        r,
        c,
        cp,
        cpp,
        z,
        xin,
        D,
        xbot,
        bot,
        bot_slope,
        debug=False
):
    """
    Integrate ray for range dependant bathymetry
    This function is intended to be called from matlab and is built to match the input and output of the matlab ray tracing code `rayintrd_bathy`.
    
    Different source depths are not supported, so y[1,:] (*or y(2,:) in matlab indexing*) must all be the same value. y[1,0] (y(2,1) in matlab indexing) is used for all ray initial conditions.

    .. note::
        If you intend to use python, you should use either {func}`pygenray.shoot_ray` or {func}`pygenray.shoot_rays`.

    Parameters
    ----------
    y0s : array_like
        array of shape (3,j) containing all ray initial conditions.
    dx : scalar
        range distance to save ray state at
    r : scalar
        integration range in meters
    c : array_like
        array of shape (m,n) specifying sound speed.
    cp : array_like
        array of shape (m,n) specifying sound speed derivate dc/dz. **not used** cp is numerically calculated from c
    cpp : array_like
        array of shape (m,n) specifying double sound speed derivate d^2c/dz^2. **not used**: ray calculation doesn't use cpp
    z : array_like
        array of shape (n,) specifying depth coordinates for sound speed arrays
    xin : array_like
        array of shape (n,) specifying range coordinates for sound speed arrays
    D : scalar
        water depth. **not currently used**
    xbot : array_like
        array of shape (k,) specifying range-dependant bathymetry ranges
    bot : array_like
        array of shape (k,) specifying range-dependant bathymetry depths
    bot_slope : array_like
        array of shape (k,) specifying range_dependant bathymetry slopes. **this array is not used in this function**. The bottom slope is calculated from xbot and bot.
    debug : bool, optional
        if True, print debug information

    Returns
    -------
    xray : np.array
        array of the ray x values
    tray : np.array
        array of the ray time values
    zray : np.array
        array of ray depth values
    pray : np.array
        array of ray parameter variables (sinθ/c)
    dzpdz : np.array
        **not calculated** None is returned
    """

    try:
        # Optional debug printing
        if debug:
            print(f"y0s type: {type(y0s)}, shape: {getattr(y0s, 'shape', 'N/A')}")
            print(f"dx type: {type(dx)}")
            print(f"r type: {type(r)}")
            print(f"c type: {type(c)}, shape: {getattr(c, 'shape', 'N/A')}")
            print(f"z type: {type(z)}, shape: {getattr(z, 'shape', 'N/A')}")
            print(f"xin type: {type(xin)}, shape: {getattr(xin, 'shape', 'N/A')}")
            print(f"D type: {type(D)}")
            print(f"xbot type: {type(xbot)}, shape: {getattr(xbot, 'shape', 'N/A')}")
            print(f"bot type: {type(bot)}, shape: {getattr(bot, 'shape', 'N/A')}")

        # Convert MATLAB array objects to proper Python types first
        y0s = convert_matlab_input(y0s)
        c = convert_matlab_input(c) 
        z = convert_matlab_input(z)
        xin = convert_matlab_input(xin)
        xbot = convert_matlab_input(xbot)
        bot = convert_matlab_input(bot)
        
        # Convert scalar inputs (these may come as 1x1 arrays from MATLAB)
        dx = convert_matlab_scalar(dx)
        r = convert_matlab_scalar(r)
        D = convert_matlab_scalar(D)

        # Convert arrays to numpy with proper shapes and types
        y0s = np.ascontiguousarray(np.asarray(y0s, dtype=np.float64))
        c = np.ascontiguousarray(np.asarray(c, dtype=np.float64))
        z = np.ascontiguousarray(np.asarray(z, dtype=np.float64)).flatten()
        xin = np.ascontiguousarray(np.asarray(xin, dtype=np.float64)).flatten()
        xbot = np.ascontiguousarray(np.asarray(xbot, dtype=np.float64)).flatten()
        bot = np.ascontiguousarray(np.asarray(bot, dtype=np.float64)).flatten()
        
        # Validate inputs
        if y0s.ndim != 2 or y0s.shape[0] != 3:
            raise ValueError(f"y0s must have shape (3, n_rays), got shape {y0s.shape}")
        
        if c.ndim != 2:
            raise ValueError(f"Sound speed array c must be 2D, got shape {c.shape}")
            
        if len(z) != c.shape[1]:
            raise ValueError(f"Depth array z length ({len(z)}) must match c columns ({c.shape[1]})")
            
        if len(xin) != c.shape[0]:
            raise ValueError(f"Range array xin length ({len(xin)}) must match c rows ({c.shape[0]})")
            
        if len(xbot) != len(bot):
            raise ValueError(f"Bathymetry arrays xbot ({len(xbot)}) and bot ({len(bot)}) must have same length")

        if debug:
            print(f"After conversion - y0s shape: {y0s.shape}, c shape: {c.shape}")
            print(f"z length: {len(z)}, xin length: {len(xin)}")
            print(f"xbot length: {len(xbot)}, bot length: {len(bot)}")
    
    except Exception as e:
        print(f"Error during input conversion: {e}")
        raise
    try:
        launch_angles = []
        # calculate ray angles
        for k in range(y0s.shape[1]):
            theta, c0 = pr.ray_angle(0, y0s[:,k], c, xin, z)
            launch_angles.append(theta)
        
        if debug:
            print(f"Calculated {len(launch_angles)} launch angles")
        
        # Construct environment
        soundspeed = xr.DataArray(c, dims=['range','depth'], coords={'range':xin, 'depth':z})
        bathymetry = xr.DataArray(bot, dims=['range'], coords={'range':xbot})
        envi = pr.OceanEnvironment2D(soundspeed, bathymetry, flat_earth_transform=False)

        # calculate x_eval
        x_eval = np.arange(0, r, dx)
        
        if debug:
            print(f"x_eval range: 0 to {r} with step {dx}, total points: {len(x_eval)}")

        # Shoot rays
        rays = pr.shoot_rays(y0s[1,0], 0, launch_angles, r, x_eval, envi)
        
        # Extract results and ensure they are proper numpy arrays
        xray = np.asarray(rays.x, dtype=np.float64)
        tray = np.asarray(rays.t, dtype=np.float64)
        zray = np.asarray(rays.z, dtype=np.float64)
        pray = np.asarray(rays.p, dtype=np.float64)
        dzpdz = None  # Not calculated
        
        if debug:
            print(f"Results - xray shape: {xray.shape}, tray shape: {tray.shape}")
            print(f"zray shape: {zray.shape}, pray shape: {pray.shape}")

        return xray, tray, zray, pray, dzpdz
        
    except Exception as e:
        print(f"Error during ray calculation: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise


def convert_matlab_input(obj):
    """Robustly convert MATLAB inputs to Python types"""
    try:
        # Handle None/empty
        if obj is None:
            return None
        
        # Handle MATLAB arrays - check for MATLAB specific types first
        obj_type_str = str(type(obj))
        
        # MATLAB arrays often have types like 'matlab.double' or similar
        if 'matlab.' in obj_type_str:
            try:
                # Try to convert using numpy directly from MATLAB array
                if hasattr(obj, '_data'):
                    # Extract the underlying data buffer
                    data = np.frombuffer(obj._data, dtype=np.float64)
                    if hasattr(obj, 'size') and len(obj.size) > 1:
                        return data.reshape(obj.size)
                    else:
                        return data
                elif hasattr(obj, 'size') and hasattr(obj, '__array_interface__'):
                    # Use array interface if available
                    return np.asarray(obj)
                else:
                    # Fallback: try to iterate and convert
                    return np.array([[float(obj[i,j]) for j in range(obj.size[1])] 
                                   for i in range(obj.size[0])] if len(obj.size) > 1 
                                   else [float(obj[i]) for i in range(obj.size[0])])
            except Exception:
                # If MATLAB-specific conversion fails, continue to other methods
                pass
        
        # Handle MATLAB double arrays (most common case)
        if hasattr(obj, '_data') and hasattr(obj, 'size'):
            # This is likely a MATLAB array object
            if hasattr(obj, 'tolist'):
                return np.array(obj.tolist())
            elif hasattr(obj, '_data'):
                return np.array(obj._data).reshape(obj.size) if hasattr(obj, 'size') else np.array(obj._data)
        
        # Handle numpy arrays or array-like objects with tolist method
        if hasattr(obj, 'tolist'):
            return np.array(obj.tolist())
        
        # Handle Python array.array objects
        if isinstance(obj, array.array):
            arr = np.array(obj)
            return arr.item() if arr.size == 1 else arr
        
        # Handle objects with __array__ method (numpy-compatible)
        if hasattr(obj, '__array__'):
            return np.asarray(obj)
        
        # Handle objects with __array_interface__ (numpy buffer protocol)
        if hasattr(obj, '__array_interface__'):
            return np.asarray(obj)
        
        # Handle other iterables (but not strings/bytes)
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                result = np.array(list(obj))
                return result
            except (ValueError, TypeError):
                pass
        
        # Handle scalar values that might be wrapped
        if hasattr(obj, 'item'):
            return obj.item()
            
        # Return as-is if we can't convert
        return obj
        
    except Exception as e:
        print(f"Warning: Could not convert MATLAB input {type(obj)}: {e}")
        # Try one more fallback - direct numpy conversion
        try:
            return np.asarray(obj)
        except:
            return obj


def convert_matlab_scalar(obj):
    """Convert MATLAB scalar input to Python scalar"""
    try:
        # Handle direct scalar types first
        if isinstance(obj, (int, float)):
            return float(obj)
        
        # Check for MATLAB scalar types
        obj_type_str = str(type(obj))
        if 'matlab.' in obj_type_str:
            try:
                # MATLAB scalars often have a simple float conversion
                return float(obj)
            except:
                pass
        
        # Convert using the general function first
        converted = convert_matlab_input(obj)
        
        # If it's an array, extract the scalar value
        if isinstance(converted, np.ndarray):
            if converted.size == 1:
                return float(converted.item())
            else:
                # If it's a 1D array with one element, still try to extract
                if converted.ndim == 1 and len(converted) == 1:
                    return float(converted[0])
                elif converted.ndim == 2 and converted.shape == (1, 1):
                    return float(converted[0, 0])
                else:
                    raise ValueError(f"Expected scalar, got array of shape {converted.shape}")
        
        # Convert to float
        return float(converted)
        
    except Exception as e:
        print(f"Error converting scalar {obj} (type: {type(obj)}): {e}")
        # Try direct conversion as fallback
        try:
            return float(obj)
        except Exception as e2:
            print(f"Direct float conversion also failed: {e2}")
            raise ValueError(f"Could not convert {obj} to scalar")


def test_matlab_conversion():
    """
    Test function to verify MATLAB input conversion works correctly.
    Call this from MATLAB as: py.pygenray.matlab_wrapper.test_matlab_conversion()
    """
    print("Testing MATLAB conversion functions...")
    
    # Test scalar conversion
    test_scalar = 3.14
    converted = convert_matlab_scalar(test_scalar)
    print(f"Scalar test: {test_scalar} -> {converted} (type: {type(converted)})")
    
    # Test array conversion
    test_array = np.array([1, 2, 3, 4, 5])
    converted = convert_matlab_input(test_array)
    print(f"Array test: shape {test_array.shape} -> shape {converted.shape}")
    
    print("Conversion tests completed successfully!")
    return True

def debug_matlab_input(obj, name="object"):
    """
    Debug function to inspect MATLAB input objects.
    Call this from MATLAB to see exactly what's being passed.
    """
    print(f"\n=== Debug info for {name} ===")
    print(f"Type: {type(obj)}")
    print(f"Type string: {str(type(obj))}")
    
    # Check for common attributes
    attrs_to_check = ['shape', 'size', 'ndim', '_data', '__array_interface__', 
                     '__array__', 'tolist', 'item', '__iter__']
    
    for attr in attrs_to_check:
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                if callable(value):
                    print(f"Has method: {attr}")
                else:
                    print(f"{attr}: {value}")
            except Exception as e:
                print(f"{attr}: <error accessing: {e}>")
    
    # Try to get more info about MATLAB objects
    if 'matlab.' in str(type(obj)):
        print("MATLAB object detected")
        try:
            print(f"MATLAB size: {obj.size if hasattr(obj, 'size') else 'N/A'}")
        except:
            pass
    
    # Try conversion
    try:
        converted = convert_matlab_input(obj)
        print(f"Conversion successful: {type(converted)}, shape: {getattr(converted, 'shape', 'N/A')}")
    except Exception as e:
        print(f"Conversion failed: {e}")
    
    print(f"=== End debug info for {name} ===\n")
    return True

def safe_rayintrd_bathy(*args, **kwargs):
    """
    A safer wrapper around rayintrd_bathy that handles MATLAB array conversion issues.
    This function tries multiple approaches to convert MATLAB arrays safely.
    """
    try:
        return rayintrd_bathy(*args, **kwargs)
    except Exception as e:
        error_msg = str(e)
        if 'PyCapsule_Import' in error_msg or 'libmwbuffer' in error_msg:
            print("Detected MATLAB array conversion issue. Trying alternative conversion...")
            
            # Convert all arguments more aggressively
            new_args = []
            for i, arg in enumerate(args):
                try:
                    if hasattr(arg, 'tolist'):
                        # Try tolist conversion for arrays
                        converted = np.array(arg.tolist())
                        new_args.append(converted)
                    elif 'matlab.' in str(type(arg)):
                        # Handle MATLAB-specific objects
                        try:
                            converted = float(arg) if np.isscalar(arg) else np.array(arg)
                            new_args.append(converted)
                        except:
                            new_args.append(arg)
                    else:
                        new_args.append(arg)
                except Exception as conv_error:
                    print(f"Warning: Could not convert argument {i}: {conv_error}")
                    new_args.append(arg)
            
            # Try again with converted arguments
            return rayintrd_bathy(*new_args, **kwargs)
        else:
            # Re-raise if it's not a MATLAB array issue
            raise

def rayintrd_bathy_simple(y0s, dx, r, c, z, xin, xbot, bot, debug=False):
    """
    Simplified wrapper for rayintrd_bathy with fewer parameters.
    This should be easier for MATLAB to call.
    """
    # Call the full function with empty arrays for unused parameters
    empty_array = np.array([])
    
    return rayintrd_bathy(
        y0s, dx, r, c, 
        empty_array,  # cp
        empty_array,  # cpp  
        z, xin, 
        0.0,  # D
        xbot, bot,
        empty_array,  # bot_slope
        debug=debug
    )


__all__ = ['rayintrd_bathy', 'safe_rayintrd_bathy', 'reload_module', 'test_matlab_conversion', 'debug_matlab_input', 'rayintrd_bathy_simple']