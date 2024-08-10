mod elements;
pub mod header;

pub use self::header::ParseHeaderError;
use self::{
    elements::ParseBoolError,
    header::{FormatHeaderError, Header, ReadHeaderError, WriteHeaderError},
};
use py_literal::Value as PyValue;
use std::{fs, io, mem};
use thiserror::Error;

/// Read an `.npy` file located at the specified path.
///
/// This is a convience function for using `File::open` followed by
/// [`ReadNpyExt::read_npy`](trait.ReadNpyExt.html#tymethod.read_npy).
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use npz::read_npy;
/// # use npz::ReadNpyError;
///
/// let arr: Array2<i32> = read_npy("resources/array.npy")?;
/// # println!("arr = {}", arr);
/// # Ok::<_, ReadNpyError>(())
/// ```
pub fn read_npy<P, T>(path: P) -> Result<T, ReadNpyError>
where
    P: AsRef<std::path::Path>,
    T: ReadNpyExt,
{
    T::read_npy(fs::File::open(path)?)
}

/// Writes an array to an `.npy` file at the specified path.
///
/// This function will create the file if it does not exist, or overwrite it if
/// it does.
///
/// This is a convenience function for `BufWriter::new(File::create(path)?)`
/// followed by [`WriteNpyExt::write_npy`].
///
/// # Example
///
/// ```no_run
/// use ndarray::array;
/// use npz::write_npy;
/// # use npz::WriteNpyError;
///
/// let arr = array![[1, 2, 3], [4, 5, 6]];
/// write_npy("array.npy", &arr)?;
/// # Ok::<_, WriteNpyError>(())
/// ```
pub fn write_npy<P, T>(path: P, array: &T) -> Result<(), WriteNpyError>
where
    P: AsRef<std::path::Path>,
    T: WriteNpyExt,
{
    array.write_npy(io::BufWriter::new(fs::File::create(path)?))
}

/// Writes an `.npy` file (sparse if possible) with bitwise-zero-filled data.
///
/// The `.npy` file represents an array with element type `A` and shape
/// specified by `shape`, with all elements of the array represented by an
/// all-zero byte-pattern. The file is written starting at the current cursor
/// location and truncated such that there are no additional bytes after the
/// `.npy` data.
///
/// This function is primarily useful for creating an `.npy` file for an array
/// larger than available memory. The file can then be memory-mapped and
/// modified using [`ViewMutNpyExt`].
///
/// # Panics
///
/// May panic if any of the following overflow `isize` or `u64`:
///
/// - the number of elements in the array
/// - the size of the array in bytes
/// - the size of the resulting file in bytes
///
/// # Considerations
///
/// ## Data is zeroed bytes
///
/// The data consists of all zeroed bytes, so this function is useful only for
/// element types for which an all-zero byte-pattern is a valid representation.
///
/// ## Sparse file
///
/// On filesystems which support [sparse files], most of the data should be
/// handled by empty blocks, i.e. not allocated on disk. If you plan to
/// memory-map the file to modify it and know that most blocks of the file will
/// ultimately contain some nonzero data, then it may be beneficial to allocate
/// space for the file on disk before modifying it in order to avoid
/// fragmentation. For example, on POSIX-like systems, you can do this by
/// calling `fallocate` on the file.
///
/// [sparse files]: https://en.wikipedia.org/wiki/Sparse_file
///
/// ## Alternatives
///
/// If all you want to do is create an array larger than the available memory
/// and don't care about actually writing the data to disk, it may be worth
/// considering alternative options:
///
/// - Add more swap space to your system, using swap file(s) if necessary, so
///   that you can allocate the array as normal.
///
/// - If you know the data will be sparse:
///
///   - Use a sparse data structure instead of `ndarray`'s array types. For
///     example, the [`sprs`](https://crates.io/crates/sprs) crate provides
///     sparse matrices.
///
///   - Rely on memory overcommitment. In other words, configure the operating
///     system to allocate more memory than actually exists. However, this risks
///     the system running out of memory if the data is not as sparse as you
///     expect.
///
/// # Example
///
/// In this example, a file containing 64 GiB of zeroed `f64` elements is
/// created. Then, an `ArrayViewMut` is created by memory-mapping the file.
/// Modifications to the data in the `ArrayViewMut` will be applied to the
/// backing file. This works even on systems with less than 64 GiB of physical
/// memory. On filesystems which support [sparse files], the disk space that's
/// actually used depends on how much data is modified.
///
/// ```no_run
/// use memmap2::MmapMut;
/// use ndarray::ArrayViewMut3;
/// use npz::{sparse_zeroed_npy, ViewMutNpyExt};
/// use std::fs::{File, OpenOptions};
///
/// let path = "array.npy";
///
/// // Create a (sparse if supported) file containing 64 GiB of zeroed data
/// let file = File::create(path)?;
/// sparse_zeroed_npy::<f64>(&file, &[1024, 2048, 4096])?;
///
/// // Memory-map the file and create the mutable view
/// let file = OpenOptions::new().read(true).write(true).open(path)?;
/// let mut mmap = unsafe { MmapMut::map_mut(&file)? };
/// let mut view_mut = ArrayViewMut3::view_mut_npy(&mut mmap)?;
///
/// // Modify an element in the view
/// view_mut[[500, 1000, 2000]] = 888.;
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub fn sparse_zeroed_npy<A>(mut file: &fs::File, shape: &[usize]) -> Result<(), WriteNpyError>
where
    A: WritableElement,
{
    use io::Seek as _;
    let shape = shape.to_vec();
    let data_bytes_len = (shape.iter().try_fold(1, |s, a| a.checked_mul(s)))
        .expect("overflow computing number of elements")
        .checked_mul(mem::size_of::<A>())
        .expect("overflow computing length of data")
        .try_into()
        .expect("overflow converting length of data to u64");
    Header {
        type_descriptor: A::type_descriptor(),
        fortran_order: false,
        shape,
    }
    .write(file)?;
    let current_offset = file.stream_position()?;
    // First, truncate the file to the current offset
    file.set_len(current_offset)?;
    // Then, zero-extend the length to represent the data (sparse if possible)
    let new_len = current_offset
        .checked_add(data_bytes_len)
        .expect("overflow computing file length");
    file.set_len(new_len)?;
    Ok(())
}

/// An array element type that can be written to an `.npy` or `.npz` file.
pub trait WritableElement: Sized {
    /// Returns a descriptor of the type that can be used in the header.
    fn type_descriptor() -> PyValue;

    /// Writes a single instance of `Self` to the writer.
    fn write<W: io::Write>(&self, writer: W) -> Result<(), WriteDataError>;

    /// Writes a slice of `Self` to the writer.
    fn write_slice<W: io::Write>(slice: &[Self], writer: W) -> Result<(), WriteDataError>;
}

/// Extension trait for writing an array to `.npy` files.
///
/// If writes are expensive (e.g. for a file or network socket) and the layout
/// of the array is not known to be in standard or Fortran layout, it is
/// strongly recommended to wrap the writer in a [`std::io::BufWriter`]. For the
/// sake of convenience, this method calls [`io::Write::flush()`] on the writer
/// before returning.
///
/// # Example
///
/// ```no_run
/// use ndarray::{array, Array2};
/// use npz::WriteNpyExt;
/// use std::{fs::File, io::BufWriter};
/// # use npz::WriteNpyError;
///
/// let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
/// let writer = BufWriter::new(File::create("array.npy")?);
/// arr.write_npy(writer)?;
/// # Ok::<_, WriteNpyError>(())
/// ```
pub trait WriteNpyExt {
    /// Writes the array to `writer` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html).
    ///
    /// This function is the Rust equivalent of
    /// [`numpy.save`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html).
    fn write_npy<W: io::Write>(&self, writer: W) -> Result<(), WriteNpyError>;
}

/// An error writing array data.
#[derive(Debug, Error)]
pub enum WriteDataError {
    /// An error caused by I/O.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

/// An error writing a `.npy` file.
#[derive(Debug, Error)]
pub enum WriteNpyError {
    /// An error caused by I/O.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// An error formatting the header.
    #[error("error formatting header: {0}")]
    FormatHeader(#[from] FormatHeaderError),
}

impl From<WriteHeaderError> for WriteNpyError {
    fn from(err: WriteHeaderError) -> Self {
        match err {
            WriteHeaderError::Io(err) => Self::Io(err),
            WriteHeaderError::Format(err) => Self::FormatHeader(err),
        }
    }
}

impl From<WriteDataError> for WriteNpyError {
    fn from(err: WriteDataError) -> Self {
        match err {
            WriteDataError::Io(err) => Self::Io(err),
        }
    }
}

/// An array element type that can be read from an `.npy` or `.npz` file.
pub trait ReadableElement: Sized {
    /// Reads to the end of the `reader`, creating a `Vec` of length `len`.
    ///
    /// This method should return `Err(_)` in at least the following cases:
    ///
    /// * if the `type_desc` does not match `Self`
    /// * if the `reader` has fewer elements than `len`
    /// * if the `reader` has extra bytes after reading `len` elements
    fn read_to_end_exact_vec<R: io::Read>(
        reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, ReadDataError>;
}

/// Extension trait for reading `Array` from `.npy` files.
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use npz::ReadNpyExt;
/// use std::fs::File;
/// # use npz::ReadNpyError;
///
/// let reader = File::open("resources/array.npy")?;
/// let arr = Array2::<i32>::read_npy(reader)?;
/// # println!("arr = {}", arr);
/// # Ok::<_, ReadNpyError>(())
/// ```
pub trait ReadNpyExt: Sized {
    /// Reads the array from `reader` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html).
    ///
    /// This function is the Rust equivalent of
    /// [`numpy.load`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html)
    /// for `.npy` files.
    fn read_npy<R: io::Read>(reader: R) -> Result<Self, ReadNpyError>;
}

/// An error reading array data.
#[derive(Debug, Error)]
pub enum ReadDataError {
    /// An error caused by I/O.
    #[error("I/O error: {0}")]
    Io(io::Error),
    /// An error parsing the booleans.
    #[error("error parsing data: {0}")]
    ParseBool(#[from] ParseBoolError),
    /// The type descriptor does not match the element type.
    #[error("incorrect descriptor ({0}) for this type")]
    WrongDescriptor(PyValue),
    /// The file does not contain all the data described in the header.
    #[error("reached EOF before reading all data")]
    MissingData,
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    #[error("file had {0} extra bytes before EOF")]
    ExtraBytes(usize),
}

impl From<io::Error> for ReadDataError {
    /// Performs the conversion.
    ///
    /// If the error kind is `UnexpectedEof`, the `MissingData` variant is
    /// returned. Otherwise, the `Io` variant is returned.
    fn from(err: io::Error) -> Self {
        match err.kind() {
            io::ErrorKind::UnexpectedEof => Self::MissingData,
            _ => Self::Io(err),
        }
    }
}

/// An error reading a `.npy` file.
#[derive(Debug, Error)]
pub enum ReadNpyError {
    /// An error caused by I/O.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// An error parsing the file header.
    #[error("error parsing header: {0}")]
    ParseHeader(#[from] ParseHeaderError),
    /// An error parsing the booleans.
    #[error("error parsing data: {0}")]
    ParseBool(ParseBoolError),
    /// Overflow while computing the length of the array (in units of bytes or
    /// the number of elements) from the shape described in the file header.
    #[error("overflow computing length from shape")]
    LengthOverflow,
    /// An error caused by incorrect `Dimension` type.
    #[error("ndim {1} of array did not match Dimension type with NDIM = {0:?}")]
    WrongNdim(Option<usize>, usize),
    /// The type descriptor does not match the element type.
    #[error("incorrect descriptor ({0}) for this type")]
    WrongDescriptor(PyValue),
    /// The file does not contain all the data described in the header.
    #[error("reached EOF before reading all data")]
    MissingData,
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    #[error("file had {0} extra bytes before EOF")]
    ExtraBytes(usize),
}

impl From<ReadHeaderError> for ReadNpyError {
    fn from(err: ReadHeaderError) -> Self {
        match err {
            ReadHeaderError::Io(err) => Self::Io(err),
            ReadHeaderError::Parse(err) => Self::ParseHeader(err),
        }
    }
}

impl From<ReadDataError> for ReadNpyError {
    fn from(err: ReadDataError) -> Self {
        match err {
            ReadDataError::Io(err) => Self::Io(err),
            ReadDataError::WrongDescriptor(desc) => Self::WrongDescriptor(desc),
            ReadDataError::MissingData => Self::MissingData,
            ReadDataError::ExtraBytes(nbytes) => Self::ExtraBytes(nbytes),
            ReadDataError::ParseBool(err) => Self::ParseBool(err),
        }
    }
}

/// An array element type that can be viewed (without copying) in an `.npy`
/// file.
pub trait ViewElement: Sized {
    /// Casts `bytes` into a slice of elements of length `len`.
    ///
    /// Returns `Err(_)` in at least the following cases:
    ///
    ///   * if the `type_desc` does not match `Self` with native endianness
    ///   * if the `bytes` slice is misaligned for elements of type `Self`
    ///   * if the `bytes` slice is too short for `len` elements
    ///   * if the `bytes` slice has extra bytes after `len` elements
    ///
    /// May panic if `len * size_of::<Self>()` overflows.
    fn bytes_as_slice<'a>(
        bytes: &'a [u8],
        type_desc: &PyValue,
        len: usize,
    ) -> Result<&'a [Self], ViewDataError>;
}

/// An array element type that can be mutably viewed (without copying) in an
/// `.npy` file.
pub trait ViewMutElement: Sized {
    /// Casts `bytes` into a mutable slice of elements of length `len`.
    ///
    /// Returns `Err(_)` in at least the following cases:
    ///
    ///   * if the `type_desc` does not match `Self` with native endianness
    ///   * if the `bytes` slice is misaligned for elements of type `Self`
    ///   * if the `bytes` slice is too short for `len` elements
    ///   * if the `bytes` slice has extra bytes after `len` elements
    ///
    /// May panic if `len * size_of::<Self>()` overflows.
    fn bytes_as_mut_slice<'a>(
        bytes: &'a mut [u8],
        type_desc: &PyValue,
        len: usize,
    ) -> Result<&'a mut [Self], ViewDataError>;
}

/// Extension trait for creating a view from a buffer containing an `.npy` file.
///
/// The primary use-case for this is viewing a memory-mapped `.npy` file.
///
/// # Notes
///
/// - For types for which not all bit patterns are valid, such as `bool`, the
///   implementation iterates over all of the elements when creating the view to
///   ensure they have a valid bit pattern.
///
/// - The data in the buffer must be properly aligned for the element type.
///   Typically, this should not be a concern for memory-mapped files (unless an
///   option like `MAP_FIXED` is used), since memory mappings are usually
///   aligned to a page boundary, and the `.npy` format has padding such that
///   the header size is a multiple of 64 bytes.
///
/// # Example
///
/// This is an example of opening a readonly memory-mapped file as an
/// `ArrayView`.
///
/// This example uses the [`memmap2`](https://crates.io/crates/memmap2) crate
/// because that appears to be the best-maintained memory-mapping crate at the
/// moment, but `view_npy` takes a `&[u8]` instead of a file so that you can
/// use the memory-mapping crate you're most comfortable with.
///
/// ```
/// # if !cfg!(miri) { // Miri doesn't support mmap.
/// use memmap2::Mmap;
/// use ndarray::ArrayView2;
/// use npz::ViewNpyExt;
/// use std::fs::File;
///
/// let file = File::open("resources/array.npy")?;
/// let mmap = unsafe { Mmap::map(&file)? };
/// let view = ArrayView2::<i32>::view_npy(&mmap)?;
/// # println!("view = {}", view);
/// # }
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub trait ViewNpyExt<'a>: Sized {
    /// Creates an `ArrayView` from a buffer containing an `.npy` file.
    fn view_npy(buf: &'a [u8]) -> Result<Self, ViewNpyError>;
}

/// Extension trait for creating a mutable view from a mutable buffer
/// containing an `.npy` file.
///
/// The primary use-case for this is modifying a memory-mapped `.npy` file.
/// Modifying the elements in the view will modify the file. Modifying the
/// shape/strides of the view will *not* modify the shape/strides of the array
/// in the file.
///
/// Notes:
///
/// - For types for which not all bit patterns are valid, such as `bool`, the
///   implementation iterates over all of the elements when creating the view to
///   ensure they have a valid bit pattern.
///
/// - The data in the buffer must be properly aligned for the element type.
///   Typically, this should not be a concern for memory-mapped files (unless an
///   option like `MAP_FIXED` is used), since memory mappings are usually
///   aligned to a page boundary, and the `.npy` format has padding such that
///   the header size is a multiple of 64 bytes.
///
/// # Example
///
/// This is an example of opening a writable memory-mapped file as a mutable
/// view. Changes to the data in the view will modify the underlying file.
///
/// This example uses the [`memmap2`](https://crates.io/crates/memmap2) crate
/// because that appears to be the best-maintained memory-mapping crate at the
/// moment, but `view_mut_npy` takes a `&mut [u8]` instead of a file so that
/// you can use the memory-mapping crate you're most comfortable with.
///
/// ```
/// # if !cfg!(miri) { // Miri doesn't support mmap.
/// use memmap2::MmapMut;
/// use ndarray::ArrayViewMut2;
/// use npz::ViewMutNpyExt;
/// use std::fs;
///
/// let file = fs::OpenOptions::new()
///     .read(true)
///     .write(true)
///     .open("resources/array.npy")?;
/// let mut mmap = unsafe { MmapMut::map_mut(&file)? };
/// let view_mut = ArrayViewMut2::<i32>::view_mut_npy(&mut mmap)?;
/// # println!("view_mut = {}", view_mut);
/// # }
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub trait ViewMutNpyExt<'a>: Sized {
    /// Creates an `ArrayViewMut` from a mutable buffer containing an `.npy`
    /// file.
    fn view_mut_npy(buf: &'a mut [u8]) -> Result<Self, ViewNpyError>;
}

/// An error viewing array data.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ViewDataError {
    /// The type descriptor does not match the element type.
    #[error("incorrect descriptor ({0}) for this type")]
    WrongDescriptor(PyValue),
    /// The type descriptor does not match the native endianness.
    #[error("descriptor does not match native endianness")]
    NonNativeEndian,
    /// The start of the data is not properly aligned for the element type.
    #[error("start of data is not properly aligned for the element type")]
    Misaligned,
    /// The file does not contain all the data described in the header.
    #[error("missing {0} bytes of data specified in header")]
    MissingBytes(usize),
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    #[error("file had {0} extra bytes before EOF")]
    ExtraBytes(usize),
    /// An error parsing the booleans.
    #[error("invalid data for element type: {0}")]
    ParseBool(#[from] ParseBoolError),
}

/// An error viewing a `.npy` file.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ViewNpyError {
    /// An error caused by I/O.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// An error parsing the file header.
    #[error("error parsing header: {0}")]
    ParseHeader(#[from] ParseHeaderError),
    /// An error parsing the booleans.
    #[error("invalid data for element type: {0}")]
    ParseBool(ParseBoolError),
    /// Overflow while computing the length of the array (in units of bytes or
    /// the number of elements) from the shape described in the file header.
    #[error("overflow computing length from shape")]
    LengthOverflow,
    /// An error caused by incorrect `Dimension` type.
    #[error("ndim {1} of array did not match Dimension type with NDIM = {0:?}")]
    WrongNdim(Option<usize>, usize),
    /// The type descriptor does not match the element type.
    #[error("incorrect descriptor ({0}) for this type")]
    WrongDescriptor(PyValue),
    /// The type descriptor does not match the native endianness.
    #[error("descriptor does not match native endianness")]
    NonNativeEndian,
    /// The start of the data is not properly aligned for the element type.
    #[error("start of data is not properly aligned for the element type")]
    MisalignedData,
    /// The file does not contain all the data described in the header.
    #[error("missing {0} bytes of data specified in header")]
    MissingBytes(usize),
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    #[error("file had {0} extra bytes before EOF")]
    ExtraBytes(usize),
}

impl From<ReadHeaderError> for ViewNpyError {
    fn from(err: ReadHeaderError) -> Self {
        match err {
            ReadHeaderError::Io(err) => Self::Io(err),
            ReadHeaderError::Parse(err) => Self::ParseHeader(err),
        }
    }
}

impl From<ViewDataError> for ViewNpyError {
    fn from(err: ViewDataError) -> Self {
        match err {
            ViewDataError::WrongDescriptor(desc) => Self::WrongDescriptor(desc),
            ViewDataError::NonNativeEndian => Self::NonNativeEndian,
            ViewDataError::Misaligned => Self::MisalignedData,
            ViewDataError::MissingBytes(nbytes) => Self::MissingBytes(nbytes),
            ViewDataError::ExtraBytes(nbytes) => Self::ExtraBytes(nbytes),
            ViewDataError::ParseBool(err) => Self::ParseBool(err),
        }
    }
}
