//! Decision strategies for a bot. The lifecycle in `BotState` is fixed; the
//! AI plugged into it is not.
//!
//! To add a new agent: define a struct implementing [`BotAgent`], add a
//! variant to [`AgentKind`], and wire it in [`AgentKind::build`]. The room
//! picks which agent to spawn via the `agent` field of the bot config.

use manabrew_agent_interface::agent_impl::Responder;
use manabrew_agent_interface::game_view_dto::GameViewDto;
use manabrew_agent_interface::prompt::{
    AgentPrompt, ChooseActionOutput, ClientToServerMessage, PromptOutput,
};
use serde::{Deserialize, Serialize};

// Native only: it reaches the arbiter over blocking HTTP, which the
// wasm build of this crate has no way to do.
#[cfg(feature = "native")]
pub mod llm;
pub mod simple_ai;
#[cfg(feature = "native")]
pub use llm::LlmAgent;
pub use simple_ai::SimpleAi;

pub trait BotAgent: Send {
    fn observe(&mut self, _view: GameViewDto) {}
    fn decide(&mut self, prompt: AgentPrompt) -> Option<PromptOutput>;
}

/// Wire-level selector for which built-in agent the bot should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum AgentKind {
    #[default]
    Simple,
    /// Every decision is asked of the MTG-LLM-Pilot arbiter over HTTP.
    /// See agent::llm -- the address comes from MANABOT_ARBITER_URL,
    /// because the model runs on a different machine from the game.
    ///
    /// Native only, like the agent behind it. A wasm build cannot reach
    /// the arbiter, and quietly seating the built-in AI instead would
    /// play a different opponent than the one that was asked for.
    #[cfg(feature = "native")]
    Llm,
}

impl AgentKind {
    pub fn build(self) -> Box<dyn BotAgent + Send> {
        match self {
            AgentKind::Simple => Box::<SimpleAi>::default(),
            #[cfg(feature = "native")]
            AgentKind::Llm => Box::<LlmAgent>::default(),
        }
    }
}

pub struct BotResponder {
    agent: Box<dyn BotAgent + Send>,
}

impl BotResponder {
    pub fn new(agent: Box<dyn BotAgent + Send>) -> Self {
        Self { agent }
    }
}

impl Default for BotResponder {
    fn default() -> Self {
        Self::new(AgentKind::default().build())
    }
}

impl BotResponder {
    /// Feed the seat's latest view to the agent, as a state envelope would.
    pub fn observe(&mut self, view: GameViewDto) {
        self.agent.observe(view);
    }
}

impl Responder for BotResponder {
    fn respond(&mut self, prompt: AgentPrompt) -> ClientToServerMessage {
        let prompt_id = prompt.prompt_id;
        let action = self
            .agent
            .decide(prompt)
            .unwrap_or(PromptOutput::ChooseAction(ChooseActionOutput::Pass {
                until: None,
                exhaust_stack: false,
            }));
        ClientToServerMessage::Response { prompt_id, action }
    }
}
