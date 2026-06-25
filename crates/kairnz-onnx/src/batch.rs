//! Batched inference: one trait, two backends (direct single-session and a
//! shared cross-thread server).

use std::sync::Mutex;

use crate::evaluator::OnnxEvaluator;

/// A batched policy/value evaluator. `planes[i]` is a canonical 14*81 plane
/// vector; `reps[i]` its repetition count. Returns one (policy, value) per row.
pub trait BatchEvaluator: Send + Sync {
    /// Evaluates a batch of pre-encoded positions, returning one (policy, value)
    /// pair per input row. Policy length is `POLICY_SIZE` (6723).
    fn evaluate_batch(
        &self,
        planes: &[Vec<f32>],
        reps: &[u8],
    ) -> ort::Result<Vec<(Vec<f32>, f32)>>;
}

/// Single-session backend (one search at a time; for the app).
pub struct DirectBatchEvaluator {
    inner: Mutex<OnnxEvaluator>,
}

impl DirectBatchEvaluator {
    /// Wraps an `OnnxEvaluator` in a `Mutex` for single-threaded batched use.
    pub fn new(evaluator: OnnxEvaluator) -> Self {
        Self {
            inner: Mutex::new(evaluator),
        }
    }
}

impl BatchEvaluator for DirectBatchEvaluator {
    fn evaluate_batch(
        &self,
        planes: &[Vec<f32>],
        reps: &[u8],
    ) -> ort::Result<Vec<(Vec<f32>, f32)>> {
        let mut guard = self.inner.lock().expect("evaluator mutex poisoned");
        guard.evaluate_batch(planes, reps)
    }
}
