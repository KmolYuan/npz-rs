use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use num_traits::ToPrimitive;
use py_literal::{
    FormatError as PyValueFormatError, ParseError as PyValueParseError, Value as PyValue,
};
use std::{convert::TryFrom, error::Error, fmt, io};

/// Magic string to indicate npy format.
const MAGIC_STRING: &[u8] = b"\x93NUMPY";

/// The total header length (including magic string, version number, header
/// length value, array format description, padding, and final newline) must be
/// evenly divisible by this value.
// If this changes, update the docs of `ViewNpyExt` and `ViewMutNpyExt`.
const HEADER_DIVISOR: usize = 64;

/// An error parsing the header of a `.npy` file.
#[derive(Debug)]
pub enum ParseHeaderError {
    /// The start of the file does not match the magic string.
    MagicString,
    /// The version number is not recognized.
    Version {
        /// Major version number.
        major: u8,
        /// Minor version number.
        minor: u8,
    },
    /// Indicates that the `HEADER_LEN` doesn't fit in `usize`.
    HeaderLengthOverflow(u32),
    /// Indicates that the array format string contains non-ASCII characters.
    /// This is an error for .npy format versions 1.0 and 2.0.
    NonAscii,
    /// Error parsing the array format string as UTF-8. This does not apply to
    /// .npy format versions 1.0 and 2.0, which require the array format string
    /// to be ASCII.
    Utf8Parse(std::str::Utf8Error),
    /// An unknown key was found in the metadata dictionary.
    UnknownKey(PyValue),
    /// A required key was missing from the metadata dictionary.
    MissingKey(&'static str),
    /// An illegal value was found for a key in the metadata dictionary.
    IllegalValue {
        /// The key for which the value was illegal.
        key: &'static str,
        /// The illegal value.
        value: PyValue,
    },
    /// Error parsing the metadata dictionary.
    DictParse(PyValueParseError),
    /// The metadata is not a dictionary.
    MetaNotDict(PyValue),
    /// The header is missing a newline at the end.
    MissingNewline,
}

impl Error for ParseHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MagicString => None,
            Self::Version { .. } => None,
            Self::HeaderLengthOverflow(_) => None,
            Self::NonAscii => None,
            Self::Utf8Parse(err) => Some(err),
            Self::UnknownKey(_) => None,
            Self::MissingKey(_) => None,
            Self::IllegalValue { .. } => None,
            Self::DictParse(err) => Some(err),
            Self::MetaNotDict(_) => None,
            Self::MissingNewline => None,
        }
    }
}

impl fmt::Display for ParseHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::MagicString => write!(f, "start does not match magic string"),
            Self::Version { major, minor } => write!(f, "unknown version number: {major}.{minor}"),
            Self::HeaderLengthOverflow(len) => write!(f, "HEADER_LEN {len} does not fit in `usize`"),
            Self::NonAscii => write!(f, "non-ascii in array format string; this is not supported in .npy format versions 1.0 and 2.0"),
            Self::Utf8Parse(err) => write!(f, "error parsing array format string as UTF-8: {err}"),
            Self::UnknownKey(key) => write!(f, "unknown key: {key}"),
            Self::MissingKey(key) => write!(f, "missing key: {key}"),
            Self::IllegalValue { key, value } => write!(f, "illegal value for key {key}: {value}"),
            Self::DictParse(err) => write!(f, "error parsing metadata dict: {err}"),
            Self::MetaNotDict(value) => write!(f, "metadata is not a dict: {value}"),
            Self::MissingNewline => write!(f, "newline missing at end of header"),
        }
    }
}

impl From<std::str::Utf8Error> for ParseHeaderError {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::Utf8Parse(err)
    }
}

impl From<PyValueParseError> for ParseHeaderError {
    fn from(err: PyValueParseError) -> Self {
        Self::DictParse(err)
    }
}

#[derive(Debug)]
pub enum ReadHeaderError {
    Io(io::Error),
    Parse(ParseHeaderError),
}

impl Error for ReadHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Parse(err) => Some(err),
        }
    }
}

impl fmt::Display for ReadHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {}", err),
            Self::Parse(err) => write!(f, "error parsing header: {}", err),
        }
    }
}

impl From<io::Error> for ReadHeaderError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<ParseHeaderError> for ReadHeaderError {
    fn from(err: ParseHeaderError) -> Self {
        Self::Parse(err)
    }
}

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
#[non_exhaustive]
enum Version {
    V1_0,
    V2_0,
    V3_0,
}

impl Version {
    /// Number of bytes taken up by version number (1 byte for major version, 1
    /// byte for minor version).
    const VERSION_NUM_BYTES: usize = 2;

    fn from_array(bytes: [u8; Self::VERSION_NUM_BYTES]) -> Result<Self, ParseHeaderError> {
        match bytes {
            [0x01, 0x00] => Ok(Version::V1_0),
            [0x02, 0x00] => Ok(Version::V2_0),
            [0x03, 0x00] => Ok(Version::V3_0),
            [major, minor] => Err(ParseHeaderError::Version { major, minor }),
        }
    }

    /// Major version number.
    const fn major_version(self) -> u8 {
        match self {
            Version::V1_0 => 1,
            Version::V2_0 => 2,
            Version::V3_0 => 3,
        }
    }

    /// Major version number.
    const fn minor_version(self) -> u8 {
        match self {
            Version::V1_0 => 0,
            Version::V2_0 => 0,
            Version::V3_0 => 0,
        }
    }

    /// Number of bytes in representation of header length.
    const fn header_len_num_bytes(self) -> usize {
        match self {
            Version::V1_0 => 2,
            Version::V2_0 | Version::V3_0 => 4,
        }
    }

    /// Read header length.
    fn read_header_len<R: io::Read>(self, mut reader: R) -> Result<usize, ReadHeaderError> {
        match self {
            Version::V1_0 => Ok(usize::from(reader.read_u16::<LittleEndian>()?)),
            Version::V2_0 | Version::V3_0 => {
                let header_len: u32 = reader.read_u32::<LittleEndian>()?;
                Ok(usize::try_from(header_len)
                    .map_err(|_| ParseHeaderError::HeaderLengthOverflow(header_len))?)
            }
        }
    }

    /// Format header length as bytes for writing to file.
    ///
    /// Returns `None` if the value of `header_len` is too large for this .npy
    /// version.
    fn format_header_len(self, header_len: usize) -> Option<Vec<u8>> {
        match self {
            Version::V1_0 => {
                let header_len = u16::try_from(header_len).ok()?;
                let mut out = vec![0; self.header_len_num_bytes()];
                LittleEndian::write_u16(&mut out, header_len);
                Some(out)
            }
            Version::V2_0 | Version::V3_0 => {
                let header_len: u32 = u32::try_from(header_len).ok()?;
                let mut out = vec![0; self.header_len_num_bytes()];
                LittleEndian::write_u32(&mut out, header_len);
                Some(out)
            }
        }
    }

    /// Computes the total header length, formatted `HEADER_LEN` value, and
    /// padding length for this .npy version.
    ///
    /// `unpadded_arr_format` is the Python literal describing the array
    /// format, formatted as an ASCII string without any padding.
    ///
    /// Returns `None` if the total header length overflows `usize` or if the
    /// value of `HEADER_LEN` is too large for this .npy version.
    fn compute_lengths(self, unpadded_arr_format: &[u8]) -> Option<HeaderLengthInfo> {
        /// Length of a '\n' char in bytes.
        const NEWLINE_LEN: usize = b"\n".len();

        let prefix_len =
            MAGIC_STRING.len() + Version::VERSION_NUM_BYTES + self.header_len_num_bytes();
        let unpadded_total_len = prefix_len
            .checked_add(unpadded_arr_format.len())?
            .checked_add(NEWLINE_LEN)?;
        let padding_len = HEADER_DIVISOR - unpadded_total_len % HEADER_DIVISOR;
        let total_len = unpadded_total_len.checked_add(padding_len)?;
        let header_len = total_len - prefix_len;
        let formatted_header_len = self.format_header_len(header_len)?;
        Some(HeaderLengthInfo { total_len, formatted_header_len })
    }
}

struct HeaderLengthInfo {
    /// Total header length (including magic string, version number, header
    /// length value, array format description, padding, and final newline).
    total_len: usize,
    /// Formatted `HEADER_LEN` value. (This is the number of bytes in the array
    /// format description, padding, and final newline.)
    formatted_header_len: Vec<u8>,
}

#[derive(Debug)]
pub enum FormatHeaderError {
    PyValue(PyValueFormatError),
    /// The total header length overflows `usize`, or `HEADER_LEN` exceeds the
    /// maximum encodable value.
    HeaderTooLong,
}

impl Error for FormatHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::PyValue(err) => Some(err),
            Self::HeaderTooLong => None,
        }
    }
}

impl fmt::Display for FormatHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::PyValue(err) => write!(f, "error formatting Python value: {err}"),
            Self::HeaderTooLong => write!(f, "the header is too long"),
        }
    }
}

impl From<PyValueFormatError> for FormatHeaderError {
    fn from(err: PyValueFormatError) -> Self {
        Self::PyValue(err)
    }
}

#[derive(Debug)]
pub enum WriteHeaderError {
    Io(io::Error),
    Format(FormatHeaderError),
}

impl Error for WriteHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Format(err) => Some(err),
        }
    }
}

impl fmt::Display for WriteHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Format(err) => write!(f, "error formatting header: {err}"),
        }
    }
}

impl From<io::Error> for WriteHeaderError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<FormatHeaderError> for WriteHeaderError {
    fn from(err: FormatHeaderError) -> Self {
        Self::Format(err)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Header {
    pub type_descriptor: PyValue,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.to_py_value().fmt(f)
    }
}

impl Header {
    fn from_py_value(value: PyValue) -> Result<Self, ParseHeaderError> {
        let PyValue::Dict(dict) = value else {
            return Err(ParseHeaderError::MetaNotDict(value));
        };
        let mut type_descriptor = None;
        let mut fortran_order = None;
        let mut shape = None;
        for (key, value) in dict {
            match &key {
                PyValue::String(k) if k == "descr" => {
                    type_descriptor = Some(value);
                }
                PyValue::String(k) if k == "fortran_order" => {
                    if let PyValue::Boolean(b) = value {
                        fortran_order = Some(b);
                    } else {
                        return Err(ParseHeaderError::IllegalValue { key: "fortran_order", value });
                    }
                }
                PyValue::String(k) if k == "shape" => {
                    fn parse_shape(value: &PyValue) -> Option<Vec<usize>> {
                        value
                            .as_tuple()?
                            .iter()
                            .map(|elem| elem.as_integer()?.to_usize())
                            .collect()
                    }
                    if let Some(s) = parse_shape(&value) {
                        shape = Some(s);
                    } else {
                        return Err(ParseHeaderError::IllegalValue { key: "shape", value });
                    }
                }
                _ => return Err(ParseHeaderError::UnknownKey(key)),
            }
        }
        let type_descriptor = type_descriptor.ok_or(ParseHeaderError::MissingKey("descr"))?;
        let fortran_order = fortran_order.ok_or(ParseHeaderError::MissingKey("fortran_order"))?;
        let shape = shape.ok_or(ParseHeaderError::MissingKey("shaper"))?;
        Ok(Self { type_descriptor, fortran_order, shape })
    }

    pub(crate) fn from_reader<R: io::Read>(mut reader: R) -> Result<Self, ReadHeaderError> {
        // Check for magic string
        {
            let mut buf = [0; MAGIC_STRING.len()];
            reader.read_exact(&mut buf)?;
            if buf != MAGIC_STRING {
                Err(ParseHeaderError::MagicString)?;
            }
        }

        // Get version number
        let mut buf = [0; Version::VERSION_NUM_BYTES];
        reader.read_exact(&mut buf)?;
        let version = Version::from_array(buf)?;

        // Get `HEADER_LEN`
        let header_len = version.read_header_len(&mut reader)?;

        // Parse the dictionary describing the array's format
        let mut buf = vec![0; header_len];
        reader.read_exact(&mut buf)?;
        let without_newline = match buf.split_last() {
            Some((&b'\n', rest)) => rest,
            Some(_) | None => Err(ParseHeaderError::MissingNewline)?,
        };
        let header_str = match version {
            Version::V1_0 | Version::V2_0 => {
                if without_newline.is_ascii() {
                    // ASCII strings are always valid UTF-8
                    unsafe { std::str::from_utf8_unchecked(without_newline) }
                } else {
                    Err(ParseHeaderError::NonAscii)?
                }
            }
            Version::V3_0 => {
                std::str::from_utf8(without_newline).map_err(ParseHeaderError::from)?
            }
        };
        let arr_format = header_str.parse().map_err(ParseHeaderError::from)?;
        Ok(Self::from_py_value(arr_format)?)
    }

    fn to_py_value(&self) -> PyValue {
        PyValue::Dict(vec![
            (
                PyValue::String("descr".to_string()),
                self.type_descriptor.clone(),
            ),
            (
                PyValue::String("fortran_order".to_string()),
                PyValue::Boolean(self.fortran_order),
            ),
            (
                PyValue::String("shape".to_string()),
                PyValue::Tuple(
                    self.shape
                        .iter()
                        .map(|&elem| PyValue::Integer(elem.into()))
                        .collect(),
                ),
            ),
        ])
    }

    fn to_bytes(&self) -> Result<Vec<u8>, FormatHeaderError> {
        // Metadata describing array's format as ASCII string
        let mut arr_format = Vec::new();
        self.to_py_value().write_ascii(&mut arr_format)?;

        // Determine appropriate version based on header length, and compute
        // length information.
        let (version, length_info) = [Version::V1_0, Version::V2_0]
            .iter()
            .find_map(|&version| Some((version, version.compute_lengths(&arr_format)?)))
            .ok_or(FormatHeaderError::HeaderTooLong)?;

        // Write the header
        let mut out = Vec::with_capacity(length_info.total_len);
        out.extend_from_slice(MAGIC_STRING);
        out.push(version.major_version());
        out.push(version.minor_version());
        out.extend_from_slice(&length_info.formatted_header_len);
        out.extend_from_slice(&arr_format);
        out.resize(length_info.total_len - 1, b' ');
        out.push(b'\n');

        // Verify the length of the header
        debug_assert_eq!(out.len(), length_info.total_len);
        debug_assert_eq!(out.len() % HEADER_DIVISOR, 0);

        Ok(out)
    }

    pub(crate) fn write<W: io::Write>(&self, mut writer: W) -> Result<(), WriteHeaderError> {
        writer.write_all(&self.to_bytes()?)?;
        Ok(())
    }
}
