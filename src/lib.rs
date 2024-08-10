#![doc = include_str!("../README.md")]
//! ## Operate .npy Files
//!
//! - Reading
//!   - [`ReadNpyExt`] extension trait
//!   - [`read_npy`] convenience function
//! - Writing
//!   - [`WriteNpyExt`] extension trait
//!   - [`write_npy`] convenience function
//!   - [`sparse_zeroed_npy`] to write an `.npy` file (sparse if possible) of
//!     zeroed data
//! - Readonly viewing (primarily for use with memory-mapped files)
//!   - [`ViewNpyExt`] extension trait
//! - Mutable viewing (primarily for use with memory-mapped files)
//!   - [`ViewMutNpyExt`] extension trait
//!
//! It's possible to create `.npy` files larger than the available memory with
//! [`sparse_zeroed_npy`] and then modify them by memory-mapping and using
//! [`ViewMutNpyExt`].
//!
//! ## Operate .npz Files
//!
//! - Reading: [`NpzReader`]
//! - Writing: [`NpzWriter`]
//!
//! ## Limitations
//!
//! - Parsing of `.npy` files is currently limited to files where the `descr`
//!   field of the [header dictionary] is a Python string literal of the form
//!   `'string'`, `"string"`, `'''string'''`, or `"""string"""`.
//!
//! - The element traits ([`WritableElement`], [`ReadableElement`],
//!   [`ViewElement`], and [`ViewMutElement`]) are currently implemented only
//!   for fixed-size integers up to 64 bits, floating point numbers, complex
//!   floating point numbers (if enabled with the crate feature), and [`bool`].
//!
//! [header dictionary]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html#format-version-1-0
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![warn(missing_docs)]

mod npy;
mod npz;

#[cfg(feature = "nalgebra")]
mod impl_nalgebra;
#[cfg(feature = "ndarray")]
mod impl_ndarray;

pub use crate::{
    npy::{
        read_npy, sparse_zeroed_npy, write_npy, ReadDataError, ReadNpyError, ReadNpyExt,
        ReadableElement, ViewDataError, ViewElement, ViewMutElement, ViewMutNpyExt, ViewNpyError,
        ViewNpyExt, WritableElement, WriteDataError, WriteNpyError, WriteNpyExt,
    },
    npz::{NpzReader, NpzWriter, ReadNpzError, WriteNpzError},
};
