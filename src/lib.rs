use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, ToPyArray, PyArray, Ix2};
use nalgebra::{DMatrix, DVector};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
struct MyStruct {
    #[pyo3(get, set)]
    value: i32,
}

#[pymethods]
impl MyStruct {
    #[new]
    fn new(value: i32) -> Self {
        MyStruct { value }
    }
    fn get_value_plus_one(&self) -> PyResult<i32> {
        Ok(self.value + 1)
    }
    fn set_value_plus_one(&mut self, value: i32) -> PyResult<()> {
        self.value = value + 1;
        Ok(())
    }
}

#[pyfunction]
fn use_my_struct(my_struct: PyRef<MyStruct>) -> String {
    format!("The value is: {}", my_struct.value)
}

// example using immutable borrows producing a new array
#[pyfunction]
fn axpy<'py>(
    py: Python<'py>,
    a: f64,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = a * &x + &y;
    z.into_pyarray_bound(py)
}

// example using a mutable borrow to modify an array in-place
#[pyfunction]
fn mult(
    a: f64, x: & PyArrayDyn<f64>
) {
    let mut x = unsafe { x.as_array_mut() };
    x *= a;
}

#[pyfunction]
fn nalgebra_dmatrix_to_numpy_ndarray<'py>(py: Python<'py>) -> Py<PyArray<f64, Ix2>> {
    // nalgebra 2x2 DMatrix
    let dmatrix = DMatrix::<f64>::from_row_slice(2, 2, &[
        1.0, 2.0,
        3.0, 4.0,
    ]);
    let numpy_array = dmatrix.to_pyarray_bound(py).unbind();
    numpy_array
}

#[pyfunction]
fn numpy_ndarray_to_nalgebra_dmatrix<'py>(x: PyReadonlyArrayDyn<'py, f64>) {
    if x.as_array().shape().len() != 2 {
        panic!("Input array must be 2D");
    }
    let nrows = x.as_array().shape()[0];
    let ncols = x.as_array().shape()[1];
    // convert to nalgebra DMatrix
    let m = match x.as_slice() {
        Ok(s) => DMatrix::<f64>::from_row_slice(nrows, ncols, s),
        Err(e) => panic!("{}", e)
    };
    println!("{}", m);
}

#[pyfunction]
fn numpy_ndarray_to_nalgebra_dvector<'py>(x: PyReadonlyArrayDyn<'py, f64>) {
    if x.as_array().shape().len() != 1 {
        panic!("Input array must be 1D");
    }
    // convert to nalgebra DVatrix
    let v= match x.as_slice() {
        Ok(s) => DVector::<f64>::from_row_slice(s),
        Err(e) => panic!("{}", e)
    };
    println!("{}", v);
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn pyo3_nalgebra_example<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<MyStruct>()?;
    m.add_function(wrap_pyfunction!(use_my_struct, m)?)?;
    m.add_function(wrap_pyfunction!(axpy, m)?)?;
    m.add_function(wrap_pyfunction!(mult, m)?)?;
    m.add_function(wrap_pyfunction!(nalgebra_dmatrix_to_numpy_ndarray, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_ndarray_to_nalgebra_dmatrix, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_ndarray_to_nalgebra_dvector, m)?)?;
    Ok(())
}
