//! Loads an ONNX model and evaluates positions into policy logits and a value.

use std::path::Path;

use kairnz_core::position::Position;
use kairnz_encode::{encode_planes, NUM_PLANES};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

use crate::Backend;

/// Board side length (9x9), matching the encoding.
const BOARD: usize = 9;

/// An ONNX model session that evaluates Kairnz positions.
pub struct OnnxEvaluator {
    session: Session,
    backend: Backend,
}

impl OnnxEvaluator {
    /// Loads a model from `path` using the CPU execution provider.
    pub fn from_path(path: &Path) -> ort::Result<OnnxEvaluator> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(OnnxEvaluator { session, backend: Backend::Cpu })
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
}
