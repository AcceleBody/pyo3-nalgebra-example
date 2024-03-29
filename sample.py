import numpy as np
import pyo3_nalgebra_example

# test Rust struct
my_struct = pyo3_nalgebra_example.MyStruct(10)
pyo3_nalgebra_example.use_my_struct(my_struct)
print(my_struct.get_value_plus_one())

# test Rust struct method
my_struct.set_value_plus_one(12)
print(my_struct.value)

# test Rust immutable arguments
a = pyo3_nalgebra_example.axpy(2.0, np.array([0.0, 1.0]), np.array([2.0, 3.0]))
print(a)

# test Rust mutable arguments
m = np.array([1.0, 2.0])
pyo3_nalgebra_example.mult(2.0, m)
print(m)

# test Rust nalgebra to Python Numpy ndarray
m = pyo3_nalgebra_example.nalgebra_dmatrix_to_numpy_ndarray()
print(m)

# test Python Numpy ndarray to Rust nalgebra
pyo3_nalgebra_example.numpy_ndarray_to_nalgebra_dmatrix(np.array([[1., 2.], [3., 4.]]))