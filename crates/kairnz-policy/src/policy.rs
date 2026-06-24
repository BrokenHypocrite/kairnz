use kairnz_core::{actions::Action, game::Game};

/// The shared interface for all Kairnz agents.
///
/// An implementor receives a `Game` reference and returns the action it
/// wishes to play, or `None` when no legal action exists.
pub trait Policy {
    /// Choose an action for the side to move in `game`, or `None` if there is
    /// no legal action.
    fn choose(&mut self, game: &Game) -> Option<Action>;

    /// Short stable identifier used in benchmark reports (e.g. `"random"`).
    fn name(&self) -> &str;
}
