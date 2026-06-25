//! Batched inference: one trait, two backends (direct single-session and a
//! shared cross-thread server).

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;

use crate::evaluator::OnnxEvaluator;

/// Default maximum positions per GPU batch.
pub const DEFAULT_MAX_BATCH: usize = 256;

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

/// One pending evaluation from a calling thread.
struct Request {
    planes: Vec<f32>,
    rep: u8,
    reply: Sender<ort::Result<(Vec<f32>, f32)>>,
}

/// A shared, thread-safe evaluator that coalesces concurrent single-position
/// requests from many self-play threads into batched GPU calls.
///
/// Internally runs a single batcher thread: it blocks on the first request,
/// then drains everything immediately available (up to `max_batch`), fires one
/// `evaluate_batch` call, and scatters results back via per-request reply
/// channels. Shuts down cleanly when dropped.
pub struct InferenceServer {
    sender: Option<Sender<Request>>,
    batcher: Option<JoinHandle<()>>,
}

impl InferenceServer {
    /// Spawns the batcher thread and returns a server backed by `evaluator`.
    pub fn new(mut evaluator: OnnxEvaluator, max_batch: usize) -> Self {
        let (tx, rx): (Sender<Request>, Receiver<Request>) = channel();
        let batcher = std::thread::spawn(move || {
            // Block for the first request, then drain everything immediately
            // available up to max_batch, run one batched inference, scatter results.
            while let Ok(first) = rx.recv() {
                let mut batch = vec![first];
                while batch.len() < max_batch {
                    match rx.try_recv() {
                        Ok(r) => batch.push(r),
                        Err(_) => break,
                    }
                }
                let planes: Vec<Vec<f32>> = batch.iter().map(|r| r.planes.clone()).collect();
                let reps: Vec<u8> = batch.iter().map(|r| r.rep).collect();
                match evaluator.evaluate_batch(&planes, &reps) {
                    Ok(results) => {
                        for (req, res) in batch.into_iter().zip(results.into_iter()) {
                            let _ = req.reply.send(Ok(res));
                        }
                    }
                    Err(e) => {
                        // Propagate the error string to every waiter in this batch.
                        let msg = e.to_string();
                        for req in batch {
                            let _ = req.reply.send(Err(ort::Error::new(msg.clone())));
                        }
                    }
                }
            }
        });
        Self { sender: Some(tx), batcher: Some(batcher) }
    }
}

impl BatchEvaluator for InferenceServer {
    /// Submits all rows to the batcher thread and waits for replies, allowing
    /// the batcher to merge these requests with other threads' work into one
    /// GPU call.
    fn evaluate_batch(
        &self,
        planes: &[Vec<f32>],
        reps: &[u8],
    ) -> ort::Result<Vec<(Vec<f32>, f32)>> {
        let sender = self.sender.as_ref().expect("server is running");
        let mut receivers = Vec::with_capacity(planes.len());
        for (p, r) in planes.iter().zip(reps.iter()) {
            let (tx, rx) = channel();
            sender
                .send(Request { planes: p.clone(), rep: *r, reply: tx })
                .map_err(|_| ort::Error::new("inference server stopped"))?;
            receivers.push(rx);
        }
        let mut out = Vec::with_capacity(receivers.len());
        for rx in receivers {
            out.push(
                rx.recv()
                    .map_err(|_| ort::Error::new("inference server dropped reply"))??,
            );
        }
        Ok(out)
    }
}

impl Drop for InferenceServer {
    fn drop(&mut self) {
        // Dropping the sender closes the channel; the batcher's recv() returns
        // Err and the thread exits cleanly.
        self.sender = None;
        if let Some(h) = self.batcher.take() {
            let _ = h.join();
        }
    }
}
