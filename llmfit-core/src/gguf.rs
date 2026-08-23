//! Minimal GGUF header reader.
//!
//! Parses the header section only (metadata KVs + tensor index) and never
//! touches tensor data, so auditing a multi-GB file costs a few KB of reads.
//! Binary layout follows gguf-py (`magic u32 "GGUF"`, `version u32`, counts
//! `u64` since v2 / `u32` in v1, typed metadata KVs, then per-tensor infos).
//!
//! Array metadata payloads are skipped after reading their type and length:
//! tokenizer vocabularies are huge and irrelevant to fit planning. Numeric
//! arrays are skipped with bounded scratch reads so the parser works on any
//! blocking reader (file today, HTTP range-streams later).
//!
//! Type ids, block sizes and metadata keys mirror ggml-org/llama.cpp
//! (`ggml.h` enum + gguf-py `GGML_QUANT_SIZES`) so weight byte counts are
//! exact rather than estimated from filenames.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DIMS: usize = 16;
const SCRATCH_SKIP_BYTES: usize = 8192;

/// ggml tensor element types (`ggml.h` enum values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2Xxs,
    IQ2Xs,
    IQ3Xxs,
    IQ1S,
    IQ4Nl,
    IQ3S,
    IQ2S,
    IQ4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    BF16,
    TQ1_0,
    TQ2_0,
    MxFp4,
    NvFp4,
    Q1_0,
    Q2_0,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2Xxs,
            17 => Self::IQ2Xs,
            18 => Self::IQ3Xxs,
            19 => Self::IQ1S,
            20 => Self::IQ4Nl,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MxFp4,
            40 => Self::NvFp4,
            41 => Self::Q1_0,
            42 => Self::Q2_0,
            other => Self::Unknown(other),
        }
    }

    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::IQ2Xxs => 16,
            Self::IQ2Xs => 17,
            Self::IQ3Xxs => 18,
            Self::IQ1S => 19,
            Self::IQ4Nl => 20,
            Self::IQ3S => 21,
            Self::IQ2S => 22,
            Self::IQ4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::IQ1M => 29,
            Self::BF16 => 30,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 35,
            Self::MxFp4 => 39,
            Self::NvFp4 => 40,
            Self::Q1_0 => 41,
            Self::Q2_0 => 42,
            Self::Unknown(id) => id,
        }
    }

    /// `(block_size, type_size_bytes)` from the ggml type traits table
    /// (gguf-py `GGML_QUANT_SIZES`). `None` for ids unknown to this build.
    pub fn layout(self) -> Option<(u64, u64)> {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => Some((1, Self::scalar_size(self))),
            Self::Q4_0 | Self::IQ4Nl => Some((32, 18)),
            Self::Q4_1 => Some((32, 20)),
            Self::Q5_0 => Some((32, 22)),
            Self::Q5_1 => Some((32, 24)),
            Self::Q8_0 => Some((32, 34)),
            Self::Q8_1 => Some((32, 36)),
            Self::MxFp4 => Some((32, 17)),
            Self::Q1_0 => Some((128, 18)),
            Self::NvFp4 => Some((64, 36)),
            Self::Q2_0 => Some((64, 18)),
            Self::Q2K => Some((256, 84)),
            Self::Q3K => Some((256, 110)),
            Self::Q4K => Some((256, 144)),
            Self::Q5K => Some((256, 176)),
            Self::Q6K => Some((256, 210)),
            Self::Q8K => Some((256, 292)),
            Self::IQ2Xxs => Some((256, 66)),
            Self::IQ2Xs => Some((256, 74)),
            Self::IQ3Xxs => Some((256, 98)),
            Self::IQ1S => Some((256, 50)),
            Self::IQ3S => Some((256, 110)),
            Self::IQ2S => Some((256, 82)),
            Self::IQ4Xs => Some((256, 136)),
            Self::IQ1M => Some((256, 56)),
            Self::TQ1_0 => Some((256, 54)),
            Self::TQ2_0 => Some((256, 66)),
            Self::Unknown(_) => None,
        }
    }

    fn scalar_size(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::I16 => 2,
            Self::I8 => 1,
            _ => 2, // F16 / BF16
        }
    }

    /// Exact stored bytes for `elements` logical elements, or `None` when the
    /// type has no layout entry in this build.
    pub fn nbytes_for(self, elements: u64) -> Option<u64> {
        self.layout()
            .map(|(block_size, type_size)| elements.div_ceil(block_size) * type_size)
    }

    /// Canonical ggml name ("Q4_K", "IQ2_XS", ...). Distinct from marketing
    /// file labels like Q4_K_M: a GGUF stores plain ggml types per tensor.
    pub fn label(self) -> String {
        match self {
            Self::Q2K => "Q2_K".into(),
            Self::Q3K => "Q3_K".into(),
            Self::Q4K => "Q4_K".into(),
            Self::Q5K => "Q5_K".into(),
            Self::Q6K => "Q6_K".into(),
            Self::Q8K => "Q8_K".into(),
            Self::IQ2Xxs => "IQ2_XXS".into(),
            Self::IQ2Xs => "IQ2_XS".into(),
            Self::IQ3Xxs => "IQ3_XXS".into(),
            Self::IQ1S => "IQ1_S".into(),
            Self::IQ4Nl => "IQ4_NL".into(),
            Self::IQ3S => "IQ3_S".into(),
            Self::IQ2S => "IQ2_S".into(),
            Self::IQ4Xs => "IQ4_XS".into(),
            Self::IQ1M => "IQ1_M".into(),
            Self::TQ1_0 => "TQ1_0".into(),
            Self::TQ2_0 => "TQ2_0".into(),
            Self::MxFp4 => "MXFP4".into(),
            Self::NvFp4 => "NVFP4".into(),
            Self::Q1_0 => "Q1_0".into(),
            Self::Q2_0 => "Q2_0".into(),
            Self::Unknown(id) => format!("UNKNOWN({id})"),
            other => format!("{other:?}").to_ascii_uppercase(),
        }
    }
}

/// Metadata value type ids from the GGUF spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    Str,
    Array,
    U64,
    I64,
    F64,
}

impl GgufValueType {
    fn from_id(id: u32) -> Result<Self, String> {
        Ok(match id {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::Str,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(format!("unknown metadata value type {other}")),
        })
    }

    fn fixed_width(self) -> Option<u64> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::Str | Self::Array => None,
        }
    }
}

/// A parsed metadata value. Array payloads are not materialized — only their
/// element type and length are kept.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array { elem_type: GgufValueType, len: u64 },
}

impl GgufValue {
    /// Unsigned view. Accepts non-negative signed ints too: some writers
    /// emit `attention.head_count` and friends as INT32.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::Bool(v) => Some(u64::from(v)),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::I8(v) => Some(i64::from(v)),
            Self::I16(v) => Some(i64::from(v)),
            Self::I32(v) => Some(i64::from(v)),
            Self::I64(v) => Some(v),
            Self::U8(v) if v <= i64::MAX as u8 => Some(i64::from(v)),
            Self::U16(v) if v <= i64::MAX as u16 => Some(i64::from(v)),
            Self::U32(v) if v <= i64::MAX as u32 => Some(i64::from(v)),
            Self::U64(v) if v <= i64::MAX as u64 => Some(v as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(f64::from(v)),
            Self::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Index of one tensor within the (unread) data section.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Logical element count (product of dimensions), saturating on overflow.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().copied().product()
    }

    /// Exact bytes this tensor occupies in the data section.
    pub fn nbytes(&self) -> Option<u64> {
        self.dtype.nbytes_for(self.n_elements())
    }
}

/// Parsed GGUF header. Tensor data is never read.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub kv_count: u64,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
}

struct Reader<R: Read> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), String> {
        self.inner
            .read_exact(buf)
            .map_err(|e| format!("truncated GGUF header: {e}"))
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        Ok(self.read_u16()? as i16)
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        Ok(self.read_u32()? as i32)
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// Consume `n` bytes without storing them (bounded scratch buffer, no
    /// seeks — keeps the parser usable over non-seekable streams).
    fn skip(&mut self, n: u64) -> Result<(), String> {
        let mut remaining = n;
        let mut scratch = [0u8; SCRATCH_SKIP_BYTES];
        while remaining > 0 {
            let chunk = remaining.min(SCRATCH_SKIP_BYTES as u64) as usize;
            self.inner
                .read_exact(&mut scratch[..chunk])
                .map_err(|e| format!("truncated GGUF header: {e}"))?;
            remaining -= chunk as u64;
        }
        Ok(())
    }

    fn read_len_prefixed_string(&mut self, wide: bool) -> Result<String, String> {
        let len = if wide {
            self.read_u64()?
        } else {
            u64::from(self.read_u32()?)
        };
        if len > MAX_STRING_BYTES {
            return Err(format!("string of {} bytes exceeds safety cap", len));
        }
        let mut bytes = vec![0u8; len as usize];
        self.read_exact(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn read_value(
        &mut self,
        value_type: GgufValueType,
        version: u32,
        depth: u32,
    ) -> Result<GgufValue, String> {
        Ok(match value_type {
            GgufValueType::U8 => GgufValue::U8(self.read_u8()?),
            GgufValueType::I8 => GgufValue::I8(self.read_u8()? as i8),
            GgufValueType::U16 => GgufValue::U16(self.read_u16()?),
            GgufValueType::I16 => GgufValue::I16(self.read_i16()?),
            GgufValueType::U32 => GgufValue::U32(self.read_u32()?),
            GgufValueType::I32 => GgufValue::I32(self.read_i32()?),
            GgufValueType::U64 => GgufValue::U64(self.read_u64()?),
            GgufValueType::I64 => GgufValue::I64(self.read_i64()?),
            GgufValueType::F32 => GgufValue::F32(self.read_f32()?),
            GgufValueType::F64 => GgufValue::F64(self.read_f64()?),
            GgufValueType::Bool => GgufValue::Bool(self.read_u8()? != 0),
            GgufValueType::Str => GgufValue::Str(self.read_len_prefixed_string(version >= 2)?),
            GgufValueType::Array => {
                if depth > 1 {
                    return Err("nested arrays in GGUF metadata are not supported".into());
                }
                let elem_type = GgufValueType::from_id(self.read_u32()?)?;
                let len = if version >= 2 {
                    self.read_u64()?
                } else {
                    u64::from(self.read_u32()?)
                };
                // Payloads are skipped; only shape is retained.
                match elem_type.fixed_width() {
                    Some(width) => self.skip(
                        width
                            .checked_mul(len)
                            .ok_or_else(|| format!("array size {}x{} overflows", width, len))?,
                    )?,
                    None if elem_type == GgufValueType::Str => {
                        for _ in 0..len {
                            let elem_len = if version >= 2 {
                                self.read_u64()?
                            } else {
                                u64::from(self.read_u32()?)
                            };
                            self.skip(elem_len)?;
                        }
                    }
                    _ => return Err("arrays of arrays in GGUF metadata are invalid".into()),
                }
                GgufValue::Array { elem_type, len }
            }
        })
    }
}

impl GgufHeader {
    /// Parse a header from any blocking reader. The reader must be positioned
    /// at offset 0 of a GGUF stream.
    pub fn read_from<R: Read>(reader: R) -> Result<Self, String> {
        let mut r = Reader { inner: reader };

        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "not a GGUF file (bad magic {magic:#010x}, expected {GGUF_MAGIC:#010x})"
            ));
        }
        let version = r.read_u32()?;
        if !(1..=3).contains(&version) {
            return Err(format!("unsupported GGUF version {version}"));
        }

        // Counts were widened to u64 in v2.
        let (tensor_count, kv_count) = if version >= 2 {
            (r.read_u64()?, r.read_u64()?)
        } else {
            (u64::from(r.read_u32()?), u64::from(r.read_u32()?))
        };

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = r.read_len_prefixed_string(version >= 2)?;
            let value_type = GgufValueType::from_id(r.read_u32()?)?;
            let value = r.read_value(value_type, version, 0)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let name = r.read_len_prefixed_string(version >= 2)?;
            let n_dims = r.read_u32()? as usize;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return Err(format!(
                    "tensor '{name}' has implausible dimension count {n_dims}"
                ));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.read_u64()?);
            }
            let dtype = GgmlType::from_id(r.read_u32()?);
            let offset = r.read_u64()?;
            tensors.push(GgufTensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }

        Ok(Self {
            version,
            kv_count,
            metadata,
            tensors,
        })
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(GgufValue::as_u64)
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(GgufValue::as_f64)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufValue::as_str)
    }

    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.metadata.get(key).and_then(GgufValue::as_i64)
    }
}

/// A GGUF file opened from disk together with its parsed header.
#[derive(Debug, Clone)]
pub struct GgufFile {
    pub path: PathBuf,
    pub file_size_bytes: u64,
    pub header: GgufHeader,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let file_size_bytes = f
            .metadata()
            .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
            .len();
        let header = GgufHeader::read_from(BufReader::new(f))?;
        Ok(Self {
            path: path.to_path_buf(),
            file_size_bytes,
            header,
        })
    }
}

// ─── Model summary ──────────────────────────────────────────────────────────

/// Architecture view derived from real GGUF metadata + tensor index.
///
/// Every field comes from the file itself — no filename heuristics.
#[derive(Debug, Clone, Serialize)]
pub struct GgufModelSummary {
    pub model_name: Option<String>,
    pub architecture: Option<String>,
    pub gguf_version: u32,

    pub block_count: Option<u64>,
    pub attention_heads: Option<u64>,
    pub key_value_heads: Option<u64>,
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub expert_feed_forward_length: Option<u64>,
    /// Multi-head Latent Attention compressed KV latent (DeepSeek family).
    pub kv_lora_rank: Option<u64>,
    pub rope_dimension_count: Option<u64>,
    pub rope_freq_base: Option<f64>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f64>,
    pub rope_original_context_length: Option<u64>,
    pub sliding_window: Option<u64>,
    pub vocab_size: Option<u64>,

    pub tensor_count: usize,
    /// Logical parameters summed over all tensors (embeddings may be tied).
    pub total_parameters: u64,
    /// Active parameters for MoE: non-expert tensors plus experts scaled by
    /// `expert_used_count / expert_count`. `None` when inputs are missing.
    pub active_parameters: Option<u64>,
    /// Exact bytes all tensors occupy on disk (excludes metadata/padding).
    pub weights_bytes: u64,
    /// Tensors whose ggml type id has no layout in this build — excluded from
    /// `weights_bytes`, listed here instead of guessed away.
    pub unknown_type_tensors: Vec<UnknownTypeTensor>,
    /// Per-type byte shares, sorted descending.
    pub quant_mix: Vec<QuantShare>,
    /// Label of the largest quant share, if any weights were recognized.
    pub dominant_quant_label: Option<String>,
    /// Merged runs of consecutive blocks sharing the same attn+ffn quant pair.
    pub layer_quants: Vec<LayerQuantRun>,
    /// Byte split by component role derived from tensor names.
    pub components: ComponentBytes,
    pub has_routed_experts: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnknownTypeTensor {
    pub name: String,
    pub type_id: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantShare {
    pub label: String,
    pub bytes: u64,
    /// Fraction of total weights bytes, 0.0–1.0.
    pub share: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerQuantRun {
    pub first_block: u32,
    pub last_block: u32,
    /// Dominant (by bytes) type among attention tensors of each block.
    pub attention: String,
    /// Dominant (by bytes) type among FFN tensors of each block, routed
    /// experts included.
    pub ffn: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ComponentBytes {
    pub embeddings: u64,
    pub output_head: u64,
    pub attention: u64,
    pub dense_ffn: u64,
    pub routed_experts: u64,
    pub other: u64,
}

fn arch_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

/// Layer index parsed from the `blk.N.` prefix llama.cpp mandates.
fn block_index(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?
        .split_once('.')
        .and_then(|(idx, _)| idx.parse().ok())
}

/// Component role inferred from standard llama.cpp tensor names.
fn component_of(name: &str) -> &'static str {
    if name.starts_with("token_embd") || name.starts_with("pos_embd") {
        "emb"
    } else if name.starts_with("output.") || name == "output.weight" {
        "out"
    } else if name.contains(".ffn_") && name.contains("_exps.") {
        "exp"
    } else if name.contains(".attn_") {
        "attn"
    } else if name.contains(".ffn_") {
        "ffn"
    } else {
        "other"
    }
}

impl GgufModelSummary {
    pub fn from_header(header: &GgufHeader) -> Self {
        let arch = header
            .get_str("general.architecture")
            .map(str::to_owned)
            .unwrap_or_default();
        let k = |suffix: &str| arch_key(&arch, suffix);

        let model_name = header
            .get_str("general.name")
            .map(str::to_owned)
            .or_else(|| header.get_str("general.basename").map(str::to_owned));

        let block_count = header.get_u64(&k("block_count"));
        let attention_heads = header.get_u64(&k("attention.head_count"));
        let key_value_heads = header.get_u64(&k("attention.head_count_kv"));

        let mut components = ComponentBytes::default();
        let mut mix: BTreeMap<GgmlType, u64> = BTreeMap::new();
        let mut total_parameters = 0u64;
        let mut weights_bytes = 0u64;
        let mut unknown_type_tensors = Vec::new();
        let mut routed_expert_elements = 0u64;
        let mut has_routed_experts = false;

        // Per-block dominant types, keyed by bytes.
        let mut attn_bytes: BTreeMap<u32, BTreeMap<GgmlType, u64>> = BTreeMap::new();
        let mut ffn_bytes: BTreeMap<u32, BTreeMap<GgmlType, u64>> = BTreeMap::new();

        for t in &header.tensors {
            let elements = t.n_elements();
            total_parameters = total_parameters.saturating_add(elements);
            if let Some(bytes) = t.nbytes() {
                weights_bytes += bytes;
                *mix.entry(t.dtype).or_default() += bytes;
                match component_of(&t.name) {
                    "emb" => components.embeddings += bytes,
                    "out" => components.output_head += bytes,
                    "exp" => components.routed_experts += bytes,
                    "attn" => components.attention += bytes,
                    "ffn" => components.dense_ffn += bytes,
                    _ => components.other += bytes,
                }
            } else if let GgmlType::Unknown(id) = t.dtype {
                unknown_type_tensors.push(UnknownTypeTensor {
                    name: t.name.clone(),
                    type_id: id,
                });
            }
            if let Some(idx) = block_index(&t.name)
                && let Some(b) = t.nbytes()
            {
                if t.name.contains(".attn_") {
                    *attn_bytes
                        .entry(idx)
                        .or_default()
                        .entry(t.dtype)
                        .or_default() += b;
                } else if t.name.contains(".ffn_") {
                    *ffn_bytes
                        .entry(idx)
                        .or_default()
                        .entry(t.dtype)
                        .or_default() += b;
                }
            }
            if t.name.contains("_exps.") {
                has_routed_experts = true;
                routed_expert_elements = routed_expert_elements.saturating_add(elements);
            }
        }

        let expert_count = header.get_u64(&k("expert_count"));
        let expert_used_count = header.get_u64(&k("expert_used_count"));
        let active_parameters = if has_routed_experts && weights_bytes > 0 {
            expert_count
                .zip(expert_used_count)
                .filter(|(count, _)| *count > 0)
                .map(|(count, used)| {
                    let inactive = routed_expert_elements
                        .saturating_mul(count.saturating_sub(used.min(count)))
                        / count;
                    total_parameters.saturating_sub(inactive)
                })
        } else if !has_routed_experts {
            Some(total_parameters)
        } else {
            None
        };

        let architecture = if arch.is_empty() {
            None
        } else {
            Some(arch.clone())
        };

        let quant_mix = {
            let mut shares: Vec<QuantShare> = mix
                .iter()
                .map(|(ty, bytes)| QuantShare {
                    label: ty.label(),
                    bytes: *bytes,
                    share: if weights_bytes > 0 {
                        *bytes as f64 / weights_bytes as f64
                    } else {
                        0.0
                    },
                })
                .collect();
            shares.sort_by_key(|share| std::cmp::Reverse(share.bytes));
            shares
        };

        let layer_quants = merge_layer_runs(&attn_bytes, &ffn_bytes);

        Self {
            model_name,
            architecture,
            gguf_version: header.version,
            block_count,
            attention_heads,
            key_value_heads,
            key_length: header.get_u64(&k("attention.key_length")),
            value_length: header.get_u64(&k("attention.value_length")),
            context_length: header.get_u64(&k("context_length")),
            embedding_length: header.get_u64(&k("embedding_length")),
            feed_forward_length: header.get_u64(&k("feed_forward_length")),
            expert_count,
            expert_used_count,
            expert_feed_forward_length: header.get_u64(&k("expert_feed_forward_length")),
            // DeepSeek MLA: converter writes rank to attention.kv_lora_rank and
            // the decoupled RoPE dim to rope.dimension_count.
            kv_lora_rank: header.get_u64(&k("attention.kv_lora_rank")),
            rope_dimension_count: header.get_u64(&k("rope.dimension_count")),
            rope_freq_base: header.get_f64(&k("rope.freq_base")),
            rope_scaling_type: header.get_str(&k("rope.scaling.type")).map(str::to_owned),
            rope_scaling_factor: header.get_f64(&k("rope.scaling.factor")),
            rope_original_context_length: header
                .get_u64(&k("rope.scaling.original_context_length")),
            sliding_window: header.get_u64(&k("attention.sliding_window")),
            vocab_size: header.get_u64(&k("vocab_size")),
            tensor_count: header.tensors.len(),
            dominant_quant_label: quant_mix.first().map(|q| q.label.clone()),
            total_parameters,
            active_parameters,
            weights_bytes,
            unknown_type_tensors,
            quant_mix,
            layer_quants,
            components,
            has_routed_experts,
        }
    }

    pub fn dominant_quant(&self) -> Option<&QuantShare> {
        self.quant_mix.first()
    }

    /// KV cache bytes consumed per token across ALL layers at fp16, following
    /// the engine formula (models.rs `kv_cache_gb`): GQA
    /// `2·L·H_kv·head_dim·dtype`, or MLA `L·(kv_lora_rank+rope_dim)·dtype`
    /// where K and V share one latent.
    pub fn kv_cache_bytes_per_token_fp16(&self) -> Result<u64, String> {
        const FP16_BYTES: u64 = 2;
        let layers = self.block_count.ok_or("block_count missing")?;
        let per_token_layer = if let Some(rank) = self.kv_lora_rank {
            rank + self.rope_dimension_count.unwrap_or(0)
        } else {
            let heads = self
                .key_value_heads
                .or(self.attention_heads)
                .ok_or("attention.head_count missing")?;
            let head_dim = self
                .key_length
                .or(self.value_length)
                .or_else(|| match (self.embedding_length, self.attention_heads) {
                    (Some(embedding), Some(heads)) if heads > 0 => Some(embedding / heads),
                    _ => None,
                })
                .ok_or("cannot determine head_dim (no key_length/embedding_length)")?;
            2 * heads * head_dim
        };
        layers
            .checked_mul(per_token_layer)
            .and_then(|elems| elems.checked_mul(FP16_BYTES))
            .ok_or_else(|| "KV size overflow".to_string())
    }

    /// KV cache footprint in GiB at `ctx_tokens` context tokens, fp16 KV.
    pub fn kv_cache_gib_at(&self, ctx_tokens: u64) -> Result<f64, String> {
        let per_token = self.kv_cache_bytes_per_token_fp16()?;
        let bytes = per_token
            .checked_mul(ctx_tokens)
            .ok_or_else(|| "KV size overflow".to_string())?;
        Ok(bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn merge_layer_runs(
    attn: &BTreeMap<u32, BTreeMap<GgmlType, u64>>,
    ffn: &BTreeMap<u32, BTreeMap<GgmlType, u64>>,
) -> Vec<LayerQuantRun> {
    fn dominant(m: &BTreeMap<GgmlType, u64>) -> String {
        m.iter()
            .max_by_key(|(_, b)| *b)
            .map_or_else(|| "-".to_string(), |(t, _)| t.label())
    }

    let mut runs: Vec<LayerQuantRun> = Vec::new();
    let indices: Vec<u32> = attn.keys().chain(ffn.keys()).copied().collect::<Vec<_>>();
    let indices = {
        let mut v = indices;
        v.sort_unstable();
        v.dedup();
        v
    };
    for idx in indices {
        let a = attn.get(&idx);
        let f = ffn.get(&idx);
        let run = LayerQuantRun {
            first_block: idx,
            last_block: idx,
            attention: a.map(dominant).unwrap_or_else(|| "-".into()),
            ffn: f.map(dominant).unwrap_or_else(|| "-".into()),
        };
        match runs.last_mut() {
            Some(prev)
                if prev.attention == run.attention
                    && prev.ffn == run.ffn
                    && prev.last_block + 1 == idx =>
            {
                prev.last_block = idx;
            }
            _ => runs.push(run),
        }
    }
    runs
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Fixture builders: hand-assembled GGUF binaries. String length prefixes
    // are u64 since v2, u32 in v1 — same rule for keys, values and tensor
    // names.

    #[derive(Default)]
    struct Fixture {
        version: u32,
        kv: Vec<u8>,
        tensors: Vec<u8>,
        n_kv: u64,
        n_tensors: u64,
    }

    impl Fixture {
        fn new(version: u32) -> Self {
            Self {
                version,
                ..Default::default()
            }
        }

        fn push_str(&mut self, out: &mut Vec<u8>, s: &str) {
            if self.version >= 2 {
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            } else {
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            }
            out.extend_from_slice(s.as_bytes());
        }

        fn push_len_prefixed(&mut self, out: &mut Vec<u8>, payload_len: usize) {
            if self.version >= 2 {
                out.extend_from_slice(&(payload_len as u64).to_le_bytes());
            } else {
                out.extend_from_slice(&(payload_len as u32).to_le_bytes());
            }
        }

        /// Key first, then value type id, then payload — the GGUF order.
        fn kv_str(mut self, key: &str, val: &str) -> Self {
            let mut buf = Vec::new();
            self.push_str(&mut buf, key);
            buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
            self.push_str(&mut buf, val);
            self.kv.extend(buf);
            self.n_kv += 1;
            self
        }

        fn kv_u32(mut self, key: &str, val: u32) -> Self {
            let mut buf = Vec::new();
            self.push_str(&mut buf, key);
            buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
            buf.extend_from_slice(&val.to_le_bytes());
            self.kv.extend(buf);
            self.n_kv += 1;
            self
        }

        fn kv_i32(mut self, key: &str, val: i32) -> Self {
            let mut buf = Vec::new();
            self.push_str(&mut buf, key);
            buf.extend_from_slice(&5u32.to_le_bytes()); // INT32
            buf.extend_from_slice(&val.to_le_bytes());
            self.kv.extend(buf);
            self.n_kv += 1;
            self
        }

        fn kv_u64(mut self, key: &str, val: u64) -> Self {
            let mut buf = Vec::new();
            self.push_str(&mut buf, key);
            buf.extend_from_slice(&10u32.to_le_bytes()); // UINT64
            buf.extend_from_slice(&val.to_le_bytes());
            self.kv.extend(buf);
            self.n_kv += 1;
            self
        }

        fn kv_f32(mut self, key: &str, val: f32) -> Self {
            let mut buf = Vec::new();
            self.push_str(&mut buf, key);
            buf.extend_from_slice(&6u32.to_le_bytes()); // FLOAT32
            buf.extend_from_slice(&val.to_le_bytes());
            self.kv.extend(buf);
            self.n_kv += 1;
            self
        }

        fn kv_arr_str(mut self, key: &str, n: u64) -> Self {
            let mut buf = Vec::new();
            self.push_str(&mut buf, key);
            buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
            buf.extend_from_slice(&8u32.to_le_bytes()); // of STRING
            buf.extend_from_slice(&n.to_le_bytes());
            self.kv.extend(buf);
            for _ in 0..n {
                let mut elem = Vec::new();
                self.push_len_prefixed(&mut elem, 3);
                elem.extend_from_slice(b"abc");
                self.kv.extend(elem);
            }
            self.n_kv += 1;
            self
        }

        fn kv_arr_u32(mut self, key: &str, vals: &[u32]) -> Self {
            let mut buf = Vec::new();
            self.push_str(&mut buf, key);
            buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
            buf.extend_from_slice(&4u32.to_le_bytes()); // of UINT32
            buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
            self.kv.extend(buf);
            for v in vals {
                self.kv.extend_from_slice(&v.to_le_bytes());
            }
            self.n_kv += 1;
            self
        }

        fn tensor(mut self, name: &str, dims: &[u64], type_id: u32) -> Self {
            let mut buf = Vec::new();
            self.push_str(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&type_id.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes()); // data offset
            self.tensors.extend(buf);
            self.n_tensors += 1;
            self
        }

        fn bytes(self) -> Vec<u8> {
            let mut out = GGUF_MAGIC.to_le_bytes().to_vec();
            out.extend_from_slice(&self.version.to_le_bytes());
            if self.version >= 2 {
                out.extend_from_slice(&self.n_tensors.to_le_bytes());
                out.extend_from_slice(&self.n_kv.to_le_bytes());
            } else {
                out.extend_from_slice(&(self.n_tensors as u32).to_le_bytes());
                out.extend_from_slice(&(self.n_kv as u32).to_le_bytes());
            }
            out.extend(self.kv);
            out.extend(self.tensors);
            out
        }

        fn parse(self) -> Result<GgufHeader, String> {
            GgufHeader::read_from(Cursor::new(self.bytes()))
        }

        fn parse_ok(self) -> GgufHeader {
            self.parse().expect("fixture should parse")
        }
    }

    /// Dense llama-like file: 2 blocks, GQA 6 heads / 2 kv heads, head_dim 128.
    fn dense_fixture(version: u32) -> Fixture {
        Fixture::new(version)
            .kv_str("general.architecture", "llama")
            .kv_str("general.name", "Tiny Llama")
            .kv_u32("llama.block_count", 2)
            .kv_u32("llama.attention.head_count", 6)
            .kv_u32("llama.attention.head_count_kv", 2)
            .kv_u32("llama.attention.key_length", 128)
            .kv_u64("llama.context_length", 4096)
            .kv_u64("llama.embedding_length", 768)
            .tensor("token_embd.weight", &[512, 768], 1) // F16
            .tensor("output.weight", &[768, 512], 14) // Q6_K
            .tensor("blk.0.attn_q.weight", &[768, 768], 8) // Q8_0
            .tensor("blk.0.ffn_down.weight", &[768, 2048], 12) // Q4_K
            .tensor("blk.1.attn_q.weight", &[768, 768], 8)
            .tensor("blk.1.ffn_down.weight", &[768, 2048], 12)
    }

    #[test]
    fn test_parses_dense_llama_header() {
        let header = dense_fixture(3).parse_ok();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensors.len(), 6);
        assert_eq!(header.get_str("general.architecture"), Some("llama"));
        assert_eq!(header.get_str("general.name"), Some("Tiny Llama"));
        assert_eq!(header.get_u64("llama.block_count"), Some(2));
        assert_eq!(header.get_u64("llama.attention.head_count_kv"), Some(2));
        assert_eq!(header.get_u64("llama.context_length"), Some(4096));

        let embd = &header.tensors[0];
        assert_eq!(embd.dtype, GgmlType::F16);
        assert_eq!(embd.n_elements(), 512 * 768);
        let attn = &header.tensors[2];
        assert_eq!(attn.dtype, GgmlType::Q8_0);

        let summary = GgufModelSummary::from_header(&header);
        assert_eq!(summary.architecture.as_deref(), Some("llama"));
        assert_eq!(summary.model_name.as_deref(), Some("Tiny Llama"));
        assert_eq!(summary.block_count, Some(2));
        assert_eq!(summary.attention_heads, Some(6));
        assert_eq!(summary.key_value_heads, Some(2));
        assert_eq!(summary.context_length, Some(4096));
        assert!(!summary.has_routed_experts);
        assert_eq!(summary.active_parameters, Some(summary.total_parameters));
    }

    #[test]
    fn test_exact_weight_bytes_and_quant_mix() {
        let header = dense_fixture(3).parse_ok();
        let summary = GgufModelSummary::from_header(&header);

        // Hand-computed tensor sizes.
        let f16_embd = 512 * 768 * 2; // F16: 1 el/block, 2 B/el
        let q6k_output = ((768 * 512) / 256) * 210; // Q6_K: 256 el/block, 210 B
        let q8_attn = ((768 * 768) / 32) * 34; // Q8_0: 32 el/block, 34 B
        let q4_ffn = ((768 * 2048) / 256) * 144; // Q4_K: 256 el/block, 144 B

        let expected = f16_embd + q6k_output + 2 * (q8_attn + q4_ffn);
        assert_eq!(summary.weights_bytes as u128, expected as u128);
        assert_eq!(
            summary.total_parameters,
            512u64 * 768 + 768 * 512 + 2 * 768 * 768 + 2 * 768 * 2048
        );

        let mix: Vec<(String, u64)> = summary
            .quant_mix
            .iter()
            .map(|s| (s.label.clone(), s.bytes))
            .collect();
        // 2x Q4_K ffn (1.77 MB) outweighs 2x Q8_0 attn (1.25 MB).
        assert_eq!(
            mix,
            vec![
                ("Q4_K".into(), 2 * q4_ffn),
                ("Q8_0".into(), 2 * q8_attn),
                ("F16".into(), f16_embd),
                ("Q6_K".into(), q6k_output),
            ]
        );
        assert_eq!(
            summary.dominant_quant().map(|q| q.label.as_str()),
            Some("Q4_K")
        );

        let share_sum: f64 = summary.quant_mix.iter().map(|s| s.share).sum();
        assert!((share_sum - 1.0).abs() < 1e-9);

        // Component split.
        assert_eq!(summary.components.embeddings, f16_embd);
        assert_eq!(summary.components.output_head, q6k_output);
        assert_eq!(summary.components.attention, 2 * q8_attn);
        assert_eq!(summary.components.dense_ffn, 2 * q4_ffn);
        assert_eq!(summary.components.routed_experts, 0);
    }

    #[test]
    fn test_gqa_kv_cache_estimate() {
        let header = dense_fixture(3).parse_ok();
        let summary = GgufModelSummary::from_header(&header);

        // per token per layer: 2 kv heads x 128 head_dim x 2 (K and V) =
        // 512 elements = 1024 bytes at fp16; two layers -> 2048 bytes/token.
        assert_eq!(summary.kv_cache_bytes_per_token_fp16().unwrap(), 2048);
        // @4096 ctx: 2048 * 4096 B = 8 MiB = 8/1024 GiB.
        let gib = summary.kv_cache_gib_at(4096).unwrap();
        assert!((gib - 8.0 / 1024.0).abs() < 1e-9);

        // Without block_count the estimate must fail loudly, not guess.
        let header_no_layers = Fixture::new(3)
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.attention.head_count", 6)
            .kv_u32("llama.attention.key_length", 128)
            .tensor("token_embd.weight", &[8, 8], 0)
            .parse_ok();
        let s = GgufModelSummary::from_header(&header_no_layers);
        assert!(s.kv_cache_bytes_per_token_fp16().is_err());
    }

    #[test]
    fn test_mla_deepseek2_kv_matches_c3_anchor() {
        // Real DeepSeek2 metadata layout: rank in attention.kv_lora_rank,
        // decoupled RoPE dim in rope.dimension_count, compressed K length =
        // rank + rope (as convert_hf_to_gguf.py writes it).
        let header = Fixture::new(3)
            .kv_str("general.architecture", "deepseek2")
            .kv_u64("deepseek2.block_count", 61)
            .kv_u64("deepseek2.attention.kv_lora_rank", 512)
            .kv_u64("deepseek2.rope.dimension_count", 64)
            .kv_u64("deepseek2.attention.key_length", 576)
            .kv_u32("deepseek2.expert_count", 160)
            .kv_u32("deepseek2.expert_used_count", 6)
            .tensor("token_embd.weight", &[1024, 7168], 12)
            .parse_ok();
        let summary = GgufModelSummary::from_header(&header);

        assert_eq!(summary.kv_lora_rank, Some(512));
        // Per token per layer: 512 + 64 = 576 shared latent elements.
        assert_eq!(
            summary.kv_cache_bytes_per_token_fp16().unwrap(),
            61 * 576 * 2
        );
        // V0-C3 anchor: R1 fp16 @32k ~= 2.15 GiB.
        let gib = summary.kv_cache_gib_at(32768).unwrap();
        let expected = (61u64 * 576 * 2 * 32768) as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!((gib - expected).abs() < 1e-9);
    }

    #[test]
    fn test_moe_active_parameters_scaled_by_used_experts() {
        let header = Fixture::new(3)
            .kv_str("general.architecture", "qwen2_moe")
            .kv_u64("qwen2_moe.block_count", 1)
            .kv_u32("qwen2_moe.expert_count", 8)
            .kv_u32("qwen2_moe.expert_used_count", 2)
            .tensor("blk.0.attn_q.weight", &[64, 64], 8) // 4096 el
            .tensor(
                "blk.0.ffn_down_exps.weight",
                &[8, 64, 256], // 131072 el routed
                12,
            )
            .parse_ok();
        let summary = GgufModelSummary::from_header(&header);

        assert!(summary.has_routed_experts);
        assert_eq!(summary.total_parameters, 4096 + 131072);
        // Inactive share: 131072 * (8-2)/8 = 98304.
        assert_eq!(
            summary.active_parameters,
            Some(4096 + 131072 - 131072 * 6 / 8)
        );
        assert_eq!(summary.components.routed_experts, (131072 / 256) * 144);
        assert_eq!(summary.components.attention, (4096 / 32) * 34);
    }

    #[test]
    fn test_layer_runs_merge_consecutive_blocks() {
        let mut f = Fixture::new(3)
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 6);
        for i in 0..4 {
            f = f
                .tensor(&format!("blk.{i}.attn_q.weight"), &[256, 256], 8)
                .tensor(&format!("blk.{i}.ffn_down.weight"), &[256, 1024], 12);
        }
        for i in 4..6 {
            f = f
                .tensor(&format!("blk.{i}.attn_q.weight"), &[256, 256], 1)
                .tensor(&format!("blk.{i}.ffn_gate_inp.weight"), &[256, 8], 1);
        }
        let header = f.parse_ok();
        let summary = GgufModelSummary::from_header(&header);

        assert_eq!(summary.layer_quants.len(), 2);
        let first = &summary.layer_quants[0];
        assert_eq!((first.first_block, first.last_block), (0, 3));
        assert_eq!(first.attention, "Q8_0");
        assert_eq!(first.ffn, "Q4_K");
        let second = &summary.layer_quants[1];
        assert_eq!((second.first_block, second.last_block), (4, 5));
        assert_eq!(second.attention, "F16");
    }

    #[test]
    fn test_array_payloads_skipped_but_shaped() {
        let header = Fixture::new(3)
            .kv_str("general.architecture", "llama")
            .kv_arr_str("tokenizer.ggml.tokens", 150_000)
            .kv_arr_str("tokenizer.ggml.merges", 250)
            .kv_arr_u32("tokenizer.ggml.token_type", &[1, 2, 3])
            .tensor("token_embd.weight", &[8, 8], 0)
            .parse_ok();

        match header.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array { elem_type, len }) => {
                assert_eq!(*elem_type, GgufValueType::Str);
                assert_eq!(*len, 150_000);
            }
            other => panic!("expected string array shape, got {other:?}"),
        }
        match header.get("tokenizer.ggml.token_type") {
            Some(GgufValue::Array { elem_type, len }) => {
                assert_eq!(*elem_type, GgufValueType::U32);
                assert_eq!(*len, 3);
            }
            other => panic!("expected u32 array shape, got {other:?}"),
        }
    }

    #[test]
    fn test_unknown_tensor_type_reported_not_guessed() {
        let header = Fixture::new(3)
            .kv_str("general.architecture", "futurearch")
            .kv_u32("futurearch.block_count", 1)
            .tensor("blk.0.attn_q.weight", &[256, 256], 200)
            .tensor("token_embd.weight", &[128, 128], 12)
            .parse_ok();
        let summary = GgufModelSummary::from_header(&header);

        assert_eq!(summary.unknown_type_tensors.len(), 1);
        assert_eq!(summary.unknown_type_tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(summary.unknown_type_tensors[0].type_id, 200);
        // Only the known tensor counts toward weights.
        assert_eq!(summary.weights_bytes, (128 * 128 / 256) * 144);
    }

    #[test]
    fn test_sliding_window_and_yarn_read() {
        let header = Fixture::new(3)
            .kv_str("general.architecture", "gemma3")
            .kv_u32("gemma3.attention.sliding_window", 1024)
            .kv_str("gemma3.rope.scaling.type", "yarn")
            .kv_f32("gemma3.rope.scaling.factor", 8.0)
            .kv_u32("gemma3.rope.scaling.original_context_length", 8192)
            .tensor("token_embd.weight", &[8, 8], 0)
            .parse_ok();
        let summary = GgufModelSummary::from_header(&header);

        assert_eq!(summary.sliding_window, Some(1024));
        assert_eq!(summary.rope_scaling_type.as_deref(), Some("yarn"));
        assert_eq!(summary.rope_scaling_factor, Some(8.0));
        assert_eq!(summary.rope_original_context_length, Some(8192));
    }

    #[test]
    fn test_v1_header_with_32bit_counts() {
        let header = dense_fixture(1).parse_ok();
        assert_eq!(header.version, 1);
        assert_eq!(header.tensors.len(), 6);
        assert_eq!(header.get_u64("llama.block_count"), Some(2));
    }

    #[test]
    fn test_rejects_bad_magic_version_and_truncation() {
        // Bad magic.
        let mut bad = vec![0xABu8; 64];
        bad[0..4].copy_from_slice("NOPE".as_bytes());
        assert!(GgufHeader::read_from(Cursor::new(bad)).is_err());

        // Unsupported version.
        let bad = Fixture::new(99).kv_str("a", "b").bytes();
        assert!(GgufHeader::read_from(Cursor::new(bad)).is_err());

        // Truncated mid-metadata: claim 10 tensors, provide none.
        let mut truncated = Fixture::new(3).bytes();
        truncated[8..16].copy_from_slice(&10u64.to_le_bytes());
        assert!(GgufHeader::read_from(Cursor::new(truncated)).is_err());

        // Empty input.
        assert!(GgufHeader::read_from(Cursor::new(Vec::new())).is_err());
    }

    #[test]
    fn test_oversized_string_is_rejected() {
        // One KV (key, then value type, then payload) declaring u64::MAX bytes.
        let mut out = GGUF_MAGIC.to_le_bytes().to_vec();
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        out.extend_from_slice(&1u64.to_le_bytes()); // kv count
        out.extend_from_slice(&12u64.to_le_bytes()); // key "general.name"
        out.extend_from_slice(b"general.name");
        out.extend_from_slice(&8u32.to_le_bytes()); // STRING
        out.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(GgufHeader::read_from(Cursor::new(out)).is_err());
    }

    #[test]
    fn test_nbytes_rounds_partial_blocks_up() {
        assert_eq!(GgmlType::Q4K.nbytes_for(300), Some(288)); // ceil(300/256)=2 blocks
        assert_eq!(GgmlType::Q4K.nbytes_for(256), Some(144));
        assert_eq!(GgmlType::F16.nbytes_for(10), Some(20));
        assert_eq!(GgmlType::F32.nbytes_for(3), Some(12));
        assert_eq!(GgmlType::Unknown(200).nbytes_for(300), None);
        assert_eq!(GgmlType::from_id(12), GgmlType::Q4K);
        assert_eq!(GgmlType::Q4K.id(), 12);
        assert_eq!(GgmlType::Unknown(42).id(), 42);
    }

    #[test]
    fn test_signed_metadata_values_are_accepted() {
        // Some writers emit head counts as INT32.
        let header = Fixture::new(3)
            .kv_str("general.architecture", "x")
            .kv_i32("x.attention.head_count", 32)
            .tensor("t.weight", &[8, 8], 0)
            .parse_ok();
        assert_eq!(header.get_u64("x.attention.head_count"), Some(32));
        assert_eq!(header.get_i64("x.attention.head_count"), Some(32));
    }
}
