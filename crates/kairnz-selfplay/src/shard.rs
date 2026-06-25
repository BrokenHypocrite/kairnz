//! Writes and reads self-play samples as `.safetensors` shards.

use std::path::Path;

use kairnz_encode::{NUM_PLANES, POLICY_SIZE};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensorError;

use crate::sample::Sample;

/// Number of board cells per plane.
const BOARD_CELLS: usize = 81;
/// Number of f32 values in one sample's planes buffer.
const PLANES_STRIDE: usize = NUM_PLANES * BOARD_CELLS;

/// Errors reading or writing a shard.
#[derive(Debug)]
pub enum ShardError {
    /// A safetensors serialization/deserialization error.
    SafeTensors(SafeTensorError),
    /// An I/O error reading the shard file.
    Io(std::io::Error),
    /// The shard's tensor shapes are inconsistent or unexpected.
    Shape(String),
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::SafeTensors(e) => write!(f, "safetensors error: {e}"),
            ShardError::Io(e) => write!(f, "I/O error reading shard: {e}"),
            ShardError::Shape(msg) => write!(f, "shard shape error: {msg}"),
        }
    }
}

impl std::error::Error for ShardError {}

impl From<SafeTensorError> for ShardError {
    fn from(e: SafeTensorError) -> Self {
        ShardError::SafeTensors(e)
    }
}

impl From<std::io::Error> for ShardError {
    fn from(e: std::io::Error) -> Self {
        ShardError::Io(e)
    }
}

/// Writes `samples` to `path` as a `.safetensors` file with tensors
/// `planes [N,14,9,9] f32`, `policy [N,6723] f32`, `value [N] f32`, and
/// `legal_mask [N,6723] u8`.
pub fn write_shard(samples: &[Sample], path: &Path) -> Result<(), ShardError> {
    let n = samples.len();

    // Flatten each field into one contiguous buffer (row-major over samples).
    let mut planes: Vec<f32> = Vec::with_capacity(n * NUM_PLANES * BOARD_CELLS);
    let mut policy: Vec<f32> = Vec::with_capacity(n * POLICY_SIZE);
    let mut value: Vec<f32> = Vec::with_capacity(n);
    let mut legal_mask: Vec<u8> = Vec::with_capacity(n * POLICY_SIZE);
    for s in samples {
        planes.extend_from_slice(&s.planes);
        policy.extend_from_slice(&s.policy);
        value.push(s.value);
        legal_mask.extend_from_slice(&s.legal_mask);
    }

    let planes_view = TensorView::new(
        Dtype::F32,
        vec![n, NUM_PLANES, 9, 9],
        bytemuck::cast_slice(&planes),
    )?;
    let policy_view =
        TensorView::new(Dtype::F32, vec![n, POLICY_SIZE], bytemuck::cast_slice(&policy))?;
    let value_view = TensorView::new(Dtype::F32, vec![n], bytemuck::cast_slice(&value))?;
    let mask_view = TensorView::new(Dtype::U8, vec![n, POLICY_SIZE], &legal_mask)?;

    safetensors::serialize_to_file(
        [
            ("planes".to_string(), planes_view),
            ("policy".to_string(), policy_view),
            ("value".to_string(), value_view),
            ("legal_mask".to_string(), mask_view),
        ],
        &None,
        path,
    )?;
    Ok(())
}

/// Loads a shard written by [`write_shard`] back into samples.
///
/// Mirrors `write_shard`'s layout exactly: `planes [N, 14, 9, 9] f32`,
/// `policy [N, 6723] f32`, `value [N] f32`, `legal_mask [N, 6723] u8`.
/// Bytes are decoded with `f32::from_le_bytes` over `chunks_exact(4)` rather
/// than `bytemuck::cast_slice`, which panics on misaligned buffers.
pub fn read_shard(path: &Path) -> Result<Vec<Sample>, ShardError> {
    let bytes = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;

    let planes_t = st.tensor("planes")?;
    let policy_t = st.tensor("policy")?;
    let value_t = st.tensor("value")?;
    let mask_t = st.tensor("legal_mask")?;

    // Derive N from the value tensor shape ([N]).
    let n = match value_t.shape() {
        [n] => *n,
        shape => {
            return Err(ShardError::Shape(format!(
                "expected value shape [N], got {shape:?}"
            )))
        }
    };

    // Validate the remaining shapes match write_shard's layout.
    let expected_planes = [n, NUM_PLANES, 9, 9];
    if planes_t.shape() != expected_planes {
        return Err(ShardError::Shape(format!(
            "expected planes shape {expected_planes:?}, got {:?}",
            planes_t.shape()
        )));
    }
    let expected_policy = [n, POLICY_SIZE];
    if policy_t.shape() != expected_policy {
        return Err(ShardError::Shape(format!(
            "expected policy shape {expected_policy:?}, got {:?}",
            policy_t.shape()
        )));
    }
    if mask_t.shape() != expected_policy {
        return Err(ShardError::Shape(format!(
            "expected legal_mask shape {expected_policy:?}, got {:?}",
            mask_t.shape()
        )));
    }

    // Decode raw little-endian bytes into f32 slices.
    let planes_f32: Vec<f32> = planes_t
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let policy_f32: Vec<f32> = policy_t
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let value_f32: Vec<f32> = value_t
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // legal_mask is u8 -- bytes are already the values.
    let mask_u8: &[u8] = mask_t.data();

    // Reconstruct one Sample per row.
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        samples.push(Sample {
            planes: planes_f32[i * PLANES_STRIDE..(i + 1) * PLANES_STRIDE].to_vec(),
            policy: policy_f32[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec(),
            value: value_f32[i],
            legal_mask: mask_u8[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec(),
        });
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Sample;

    fn tiny_sample(value: f32) -> Sample {
        Sample {
            planes: vec![0.0; NUM_PLANES * BOARD_CELLS],
            policy: vec![0.0; POLICY_SIZE],
            value,
            legal_mask: vec![1u8; POLICY_SIZE],
        }
    }

    #[test]
    fn write_shard_roundtrips_shapes_and_values() {
        let samples = vec![tiny_sample(1.0), tiny_sample(-1.0)];
        let dir = std::env::temp_dir();
        let path = dir.join(format!("kairnz_selfplay_test_shard_{}.safetensors", std::process::id()));
        write_shard(&samples, &path).expect("shard writes");

        let bytes = std::fs::read(&path).expect("read shard");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("deserialize");

        let planes = st.tensor("planes").expect("planes tensor");
        assert_eq!(planes.shape(), &[2, NUM_PLANES, 9, 9]);
        assert_eq!(planes.dtype(), Dtype::F32);

        let value = st.tensor("value").expect("value tensor");
        assert_eq!(value.shape(), &[2]);
        // Decode the two f32 values from the raw little-endian bytes. Decode via
        // from_le_bytes rather than bytemuck::cast_slice, because the file buffer
        // is not guaranteed to be 4-byte aligned (cast_slice would panic).
        let decoded: Vec<f32> = value
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, vec![1.0, -1.0]);

        let mask = st.tensor("legal_mask").expect("mask tensor");
        assert_eq!(mask.dtype(), Dtype::U8);
        assert_eq!(mask.shape(), &[2, POLICY_SIZE]);

        let policy = st.tensor("policy").expect("policy tensor");
        assert_eq!(policy.shape(), &[2, POLICY_SIZE]);
        assert_eq!(policy.dtype(), Dtype::F32);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_shard_roundtrips_write_shard() {
        // Build distinct samples so byte-equality catches any field mix-up.
        let mut s0 = tiny_sample(0.5);
        s0.planes[0] = 1.0;
        s0.policy[1] = 0.75;
        s0.legal_mask[2] = 0;

        let mut s1 = tiny_sample(-0.5);
        s1.planes[NUM_PLANES * BOARD_CELLS - 1] = 2.0;
        s1.policy[POLICY_SIZE - 1] = 0.25;

        let originals = vec![s0, s1];

        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "kairnz_selfplay_roundtrip_{}.safetensors",
            std::process::id()
        ));
        write_shard(&originals, &path).expect("write_shard succeeds");
        let recovered = read_shard(&path).expect("read_shard succeeds");
        let _ = std::fs::remove_file(&path);

        assert_eq!(recovered.len(), originals.len(), "same number of samples");
        for (i, (got, want)) in recovered.iter().zip(originals.iter()).enumerate() {
            assert_eq!(got.planes, want.planes, "sample {i} planes mismatch");
            assert_eq!(got.policy, want.policy, "sample {i} policy mismatch");
            assert_eq!(got.value, want.value, "sample {i} value mismatch");
            assert_eq!(got.legal_mask, want.legal_mask, "sample {i} legal_mask mismatch");
        }
    }
}
