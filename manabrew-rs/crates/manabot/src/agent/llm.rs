//! A BotAgent that asks the MTG-LLM-Pilot arbiter what to do.
//!
//! Deploy to manabrew-rs/crates/manabot/src/agent/llm.rs.
//!
//! The bot joins a relay room as an ordinary client (the relay cannot tell
//! it from a person) and answers every prompt addressed to its seat. All
//! this file does is forward the prompt, plus the most recent board state,
//! to the arbiter's /decide endpoint and hand back what it returns. Every
//! judgement -- what the board says, how the question is phrased, which
//! answers are legal -- stays in the Python arbiter, where it is already
//! built and tested.
//!
//! WHY THE BOARD IS CARRIED SEPARATELY. `BotAgent::decide` receives only
//! the prompt, and the arbiter needs the board to describe it. The relay
//! does send the board, as `StateEnvelope::State`. `BotAgent::observe` is
//! a default-empty hook so SimpleAi, which never needed a board, is
//! untouched, and this agent keeps the latest view to send alongside the
//! prompt.
//!
//! BLOCKING ON PURPOSE, AND THE RISK. `decide` is synchronous and is
//! called inline from the websocket read loop, so this HTTP call blocks
//! that task for as long as the model takes -- measured at 7-15s per
//! decision on the 14B. Frames are not read during that window. The bot
//! already delays its answers deliberately (`BotState::answer_delay`), so
//! slow answers are anticipated, but a relay with a short ping timeout may
//! not tolerate 15 seconds. If a bot starts dropping its connection
//! mid-game, this is the first place to look: the fix is to move the call
//! off the read loop, not to make the model faster.

use std::time::Duration;

use manabrew_agent_interface::game_view_dto::GameViewDto;
use manabrew_agent_interface::prompt::{AgentPrompt, PromptOutput};
use serde_json::{json, Value};

use super::BotAgent;

/// Where the arbiter is listening. An env var rather than a constant: the
/// model lives on a different machine from the game in the setup this was
/// written for, and that address is a deployment fact, not a code one.
const ARBITER_URL_ENV: &str = "MANABOT_ARBITER_URL";
const DEFAULT_ARBITER_URL: &str = "http://127.0.0.1:8090";

/// Generous, because it is bounded by the model rather than the network:
/// a decision that needs a retry runs the model up to four times, and the
/// arbiter's own per-request ceiling is 1200s. Timing out here while the
/// arbiter is still working would produce no answer AND waste the work.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(1800);

pub struct LlmAgent {
    url: String,
    /// The most recent board state for this seat, from
    /// `StateEnvelope::State`. `None` until the first one arrives -- the
    /// arbiter is sent an empty board rather than nothing, so it can still
    /// answer prompts that do not depend on the board (a dice roll, a
    /// mulligan) instead of the bot stalling before the game starts.
    last_state: Option<Value>,
}

impl Default for LlmAgent {
    fn default() -> Self {
        Self {
            url: std::env::var(ARBITER_URL_ENV).unwrap_or_else(|_| DEFAULT_ARBITER_URL.to_string()),
            last_state: None,
        }
    }
}

impl LlmAgent {
    pub fn new() -> Self {
        Self::default()
    }

    /// "player-3" -> 3. The arbiter addresses seats by index because every
    /// deck list and identity it holds is indexed; the wire uses the id.
    fn seat_of(prompt: &AgentPrompt) -> usize {
        prompt
            .deciding_player_id
            .rsplit('-')
            .next()
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(0)
    }
}

impl BotAgent for LlmAgent {
    /// The board for this seat, as JSON, because JSON is what goes to the
    /// arbiter. `GameViewDto` is the wire shape -- the same type
    /// `java_backend` parses a Forge snapshot into -- so the round trip
    /// loses nothing the arbiter reads.
    ///
    /// Only overwritten on success: a serialisation failure must not
    /// replace a good board with nothing and send the model back to
    /// deciding blind.
    fn observe(&mut self, view: GameViewDto) {
        if let Ok(value) = serde_json::to_value(&view) {
            self.last_state = Some(value);
        }
    }

    fn decide(&mut self, prompt: AgentPrompt) -> Option<PromptOutput> {
        let seat = Self::seat_of(&prompt);
        let body = json!({
            "prompt": prompt,
            "snapshot": self.last_state.clone().unwrap_or_else(|| json!({})),
            // The arbiter fills in what it can when the deck lists are not
            // known here -- the bot is told which decks are in play by the
            // room, not by us.
            "players": Value::Null,
            "deciding_seat": seat,
        });

        let endpoint = format!("{}/decide", self.url.trim_end_matches('/'));
        let response = ureq::post(&endpoint)
            .timeout(REQUEST_TIMEOUT)
            .send_json(body);

        match response {
            Ok(resp) => match resp.into_json::<Value>() {
                Ok(value) => match serde_json::from_value::<PromptOutput>(
                    value.get("output").cloned().unwrap_or(Value::Null),
                ) {
                    Ok(output) => Some(output),
                    Err(error) => {
                        // Returning None makes BotState drop the prompt,
                        // which stalls that seat rather than playing a
                        // guess. For a game a person is sitting in front
                        // of, a visible stall beats a silent wrong move.
                        tracing::error!(target: "llm-bot",
                            "arbiter output did not parse as PromptOutput: {error}; raw={value}");
                        None
                    }
                },
                Err(error) => {
                    tracing::error!(target: "llm-bot", "arbiter reply was not JSON: {error}");
                    None
                }
            },
            Err(error) => {
                tracing::error!(target: "llm-bot", "arbiter call failed: {error}");
                None
            }
        }
    }
}
