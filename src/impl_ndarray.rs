use crate::{
    npy::header::Header, ReadNpyError, ReadNpyExt, ReadableElement, ViewElement, ViewMutElement,
    ViewMutNpyExt, ViewNpyError, ViewNpyExt, WritableElement, WriteNpyError, WriteNpyExt,
};
use ndarray::{prelude::*, Data, DataOwned, IntoDimension as _};
use std::{io, mem};

impl<A, S, D> WriteNpyExt for ArrayBase<S, D>
where
    A: WritableElement,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn write_npy<W: io::Write>(&self, mut writer: W) -> Result<(), WriteNpyError> {
        let write_contiguous = |mut writer: W, fortran_order: bool| {
            Header {
                type_descriptor: A::type_descriptor(),
                fortran_order,
                shape: self.shape().to_owned(),
            }
            .write(&mut writer)?;
            A::write_slice(self.as_slice_memory_order().unwrap(), &mut writer)?;
            writer.flush()?;
            Ok(())
        };
        if self.is_standard_layout() {
            write_contiguous(writer, false)
        } else if self.view().reversed_axes().is_standard_layout() {
            write_contiguous(writer, true)
        } else {
            Header {
                type_descriptor: A::type_descriptor(),
                fortran_order: false,
                shape: self.shape().to_owned(),
            }
            .write(&mut writer)?;
            for elem in self.iter() {
                elem.write(&mut writer)?;
            }
            writer.flush()?;
            Ok(())
        }
    }
}

impl<A, S, D> ReadNpyExt for ArrayBase<S, D>
where
    A: ReadableElement,
    S: DataOwned<Elem = A>,
    D: Dimension,
{
    fn read_npy<R: io::Read>(mut reader: R) -> Result<Self, ReadNpyError> {
        let header = Header::from_reader(&mut reader)?;
        let shape = header.shape.into_dimension();
        let ndim = shape.ndim();
        let len = shape_length_checked::<A>(&shape).ok_or(ReadNpyError::LengthOverflow)?;
        let data = A::read_to_end_exact_vec(&mut reader, &header.type_descriptor, len)?;
        ArrayBase::from_shape_vec(shape.set_f(header.fortran_order), data)
            .unwrap()
            .into_dimensionality()
            .map_err(|_| ReadNpyError::WrongNdim(D::NDIM, ndim))
    }
}

impl<'a, A, D> ViewNpyExt<'a> for ArrayView<'a, A, D>
where
    A: ViewElement,
    D: Dimension,
{
    fn view_npy(buf: &'a [u8]) -> Result<Self, ViewNpyError> {
        let mut reader = buf;
        let header = Header::from_reader(&mut reader)?;
        let shape = header.shape.into_dimension();
        let ndim = shape.ndim();
        let len = shape_length_checked::<A>(&shape).ok_or(ViewNpyError::LengthOverflow)?;
        let data = A::bytes_as_slice(reader, &header.type_descriptor, len)?;
        ArrayView::from_shape(shape.set_f(header.fortran_order), data)
            .unwrap()
            .into_dimensionality()
            .map_err(|_| ViewNpyError::WrongNdim(D::NDIM, ndim))
    }
}

impl<'a, A, D> ViewMutNpyExt<'a> for ArrayViewMut<'a, A, D>
where
    A: ViewMutElement,
    D: Dimension,
{
    fn view_mut_npy(buf: &'a mut [u8]) -> Result<Self, ViewNpyError> {
        let mut reader = &*buf;
        let header = Header::from_reader(&mut reader)?;
        let shape = header.shape.into_dimension();
        let ndim = shape.ndim();
        let len = shape_length_checked::<A>(&shape).ok_or(ViewNpyError::LengthOverflow)?;
        let mid = buf.len() - reader.len();
        let data = A::bytes_as_mut_slice(&mut buf[mid..], &header.type_descriptor, len)?;
        ArrayViewMut::from_shape(shape.set_f(header.fortran_order), data)
            .unwrap()
            .into_dimensionality()
            .map_err(|_| ViewNpyError::WrongNdim(D::NDIM, ndim))
    }
}

/// Computes the length associated with the shape (i.e. the product of the axis
/// lengths), where the element type is `T`.
///
/// Returns `None` if the number of elements or the length in bytes would
/// overflow `isize`.
fn shape_length_checked<A>(shape: &IxDyn) -> Option<usize> {
    const MAX: usize = isize::MAX as usize;
    let len = shape.size_checked()?;
    (len.checked_mul(mem::size_of::<A>())? < MAX).then_some(len)
}
