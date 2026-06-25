//! Loads an ONNX model and evaluates positions into policy logits and a value.

use std::path::Path;

use kairnz_core::position::Position;
use kairnz_encode::{encode_planes, CH_REPETITION, NUM_PLANES, POLICY_SIZE, REPETITION_NORM};
use ndarray::Array4;
use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
use ort::session::Session;
use ort::value::Tensor;

use crate::Backend;

/// Board side length (9x9), matching the encoding.
const BOARD: usize = 9;

/// Flat length of one encoded position: NUM_PLANES channels * 9*9 cells.
pub(crate) const PLANE_LEN: usize = NUM_PLANES * BOARD * BOARD;

/// An ONNX model session that evaluates Kairnz positions.
pub struct OnnxEvaluator {
    session: Session,
    backend: Backend,
}

impl OnnxEvaluator {
    /// Loads a model from `path`, attempting the CUDA execution provider and
    /// falling back to CPU. The chosen backend is recorded and reported by
    /// [`OnnxEvaluator::backend`]. CUDA failures are non-fatal.
    pub fn from_path(path: &Path) -> ort::Result<OnnxEvaluator> {
        let mut builder = Session::builder()?;
        let cuda = CUDAExecutionProvider::default();
        let backend = if cuda.register(&mut builder).is_ok() {
            Backend::Cuda
        } else {
            Backend::Cpu
        };
        let session = builder.commit_from_file(path)?;
        Ok(OnnxEvaluator { session, backend })
    }

    /// Returns the execution backend this session is running on.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Evaluates `pos`, returning the policy logits (length `POLICY_SIZE`) and
    /// the scalar value in `[-1, 1]`. `repetition_count` is the encoder input
    /// described in the encoding contract (0 when no history is tracked).
    pub fn evaluate(
        &mut self,
        pos: &Position,
        repetition_count: u8,
    ) -> ort::Result<(Vec<f32>, f32)> {
        let planes = encode_planes(pos, repetition_count);
        let input = Array4::from_shape_vec((1, NUM_PLANES, BOARD, BOARD), planes)
            .expect("encode_planes returns NUM_PLANES * 81 elements");

        let outputs = self
            .session
            .run(ort::inputs!["planes" => Tensor::from_array(input)?])?;

        let (_p_shape, policy) = outputs["policy"].try_extract_tensor::<f32>()?;
        let (_v_shape, value) = outputs["value"].try_extract_tensor::<f32>()?;

        Ok((policy.to_vec(), value[0]))
    }

    /// Evaluates a batch of pre-encoded positions in a single inference call.
    ///
    /// `planes[i]` must be a canonical plane vector of length `PLANE_LEN`
    /// (14 channels * 81 cells = 1134 floats). `reps[i]` is the repetition
    /// count for position `i`; it is applied by overwriting the repetition
    /// channel (channel 13) of each row, mirroring what `encode_planes` does
    /// internally. Returns one `(policy, value)` pair per input row, where
    /// policy has length `POLICY_SIZE` (6723).
    pub fn evaluate_batch(
        &mut self,
        planes: &[Vec<f32>],
        reps: &[u8],
    ) -> ort::Result<Vec<(Vec<f32>, f32)>> {
        assert_eq!(
            planes.len(),
            reps.len(),
            "planes and reps must have the same length"
        );
        let batch = planes.len();
        if batch == 0 {
            return Ok(Vec::new());
        }

        // Build a flat [B * PLANE_LEN] buffer with the repetition channel
        // overwritten per row, then reshape to [B, 14, 9, 9].
        let cells = BOARD * BOARD;
        let rep_offset = CH_REPETITION * cells;
        let mut flat: Vec<f32> = Vec::with_capacity(batch * PLANE_LEN);
        for (row_planes, &rep) in planes.iter().zip(reps.iter()) {
            assert_eq!(
                row_planes.len(),
                PLANE_LEN,
                "each plane vector must have length PLANE_LEN (14*81)"
            );
            let rep_val = rep as f32 / REPETITION_NORM;
            flat.extend_from_slice(row_planes);
            // Overwrite the repetition channel in the just-appended row.
            let row_start = (flat.len() - PLANE_LEN) + rep_offset;
            for cell in 0..cells {
                flat[row_start + cell] = rep_val;
            }
        }

        let input = Array4::from_shape_vec((batch, NUM_PLANES, BOARD, BOARD), flat)
            .expect("flat has batch * NUM_PLANES * 81 elements");

        let outputs = self
            .session
            .run(ort::inputs!["planes" => Tensor::from_array(input)?])?;

        let (_p_shape, policy_view) = outputs["policy"].try_extract_tensor::<f32>()?;
        let (_v_shape, value_view) = outputs["value"].try_extract_tensor::<f32>()?;

        // Collect the full flat output arrays then slice per row.
        let policy_flat: Vec<f32> = policy_view.iter().copied().collect();
        let value_flat: Vec<f32> = value_view.iter().copied().collect();

        let results = (0..batch)
            .map(|i| {
                let p_start = i * POLICY_SIZE;
                let policy = policy_flat[p_start..p_start + POLICY_SIZE].to_vec();
                let value = value_flat[i];
                (policy, value)
            })
            .collect();

        Ok(results)
    }
}
