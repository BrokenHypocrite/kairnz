//! Writes self-play samples to a `.safetensors` shard.

use std::path::Path;

use kairnz_encode::{NUM_PLANES, POLICY_SIZE};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensorError;

use crate::sample::Sample;

/// Number of board cells per plane.
const BOARD_CELLS: usize = 81;

/// Errors writing a shard.
#[derive(Debug)]
pub enum ShardError {
    /// A safetensors serialization error.
    SafeTensors(SafeTensorError),
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::SafeTensors(e) => write!(f, "safetensors error: {e}"),
        }
    }
}

impl std::error::Error for ShardError {}

impl From<SafeTensorError> for ShardError {
    fn from(e: SafeTensorError) -> Self {
        ShardError::SafeTensors(e)
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
        let path = dir.join("kairnz_selfplay_test_shard.safetensors");
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

        let _ = std::fs::remove_file(&path);
    }
}
