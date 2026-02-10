use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsStreamEvent {
    pub id: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatCompletionsStreamChoice>,
    #[serde(default)]
    pub usage: Option<ChatCompletionsUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsStreamChoice {
    pub delta: ChatCompletionsDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub index: usize,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChatCompletionsToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsToolCallDelta {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default, rename = "type")]
    pub r#type: Option<String>,
    #[serde(default)]
    pub function: Option<ChatCompletionsFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsUsage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

impl From<ChatCompletionsUsage> for TokenUsage {
    fn from(value: ChatCompletionsUsage) -> Self {
        TokenUsage {
            input_tokens: value.prompt_tokens,
            cached_input_tokens: 0,
            output_tokens: value.completion_tokens,
            reasoning_output_tokens: 0,
            total_tokens: value.total_tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub choices: Vec<ChatCompletionsResponseChoice>,
    #[serde(default)]
    pub usage: Option<ChatCompletionsUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsResponseChoice {
    pub message: ChatCompletionsResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsResponseMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChatCompletionsToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub r#type: Option<String>,
    pub function: ChatCompletionsFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsFunctionCall {
    pub name: String,
    pub arguments: String,
}

pub fn spawn_chat_completions_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) -> ResponseStream {
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(process_sse(
        stream_response.bytes,
        tx_event,
        idle_timeout,
        telemetry,
    ));
    ResponseStream { rx_event }
}

pub fn parse_chat_completions_response(body: &[u8]) -> Result<ChatCompletionsResponse, ApiError> {
    serde_json::from_slice(body).map_err(|err| {
        ApiError::Stream(format!(
            "failed to decode chat completions response: {err}; body: {}",
            String::from_utf8_lossy(body)
        ))
    })
}

pub fn chat_completions_response_to_events(
    response: ChatCompletionsResponse,
) -> Vec<ResponseEvent> {
    let mut events = Vec::new();
    if let Some(choice) = response.choices.first() {
        if let Some(content) = choice.message.content.as_ref()
            && !content.is_empty()
        {
            let item = ResponseItem::Message {
                id: None,
                role: choice.message.role.clone(),
                content: vec![ContentItem::OutputText {
                    text: content.clone(),
                }],
                end_turn: None,
                phase: None,
            };
            events.push(ResponseEvent::OutputItemDone(item));
        }

        if let Some(tool_calls) = choice.message.tool_calls.as_ref() {
            for (index, tool_call) in tool_calls.iter().enumerate() {
                let call_id = tool_call
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("tool_call_{index}"));
                let item = ResponseItem::FunctionCall {
                    id: None,
                    name: tool_call.function.name.clone(),
                    arguments: tool_call.function.arguments.clone(),
                    call_id,
                };
                events.push(ResponseEvent::OutputItemDone(item));
            }
        }
    }

    events.push(ResponseEvent::Completed {
        response_id: response.id,
        token_usage: response.usage.map(Into::into),
    });
    events
}

#[derive(Default)]
struct ToolCallAccumulator {
    calls: Vec<PartialToolCall>,
    by_id: HashMap<String, usize>,
    by_index: HashMap<u32, usize>,
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    index: Option<u32>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAccumulator {
    fn apply_delta(&mut self, delta: &ChatCompletionsToolCallDelta) {
        let entry = match (delta.id.as_ref(), delta.index) {
            (Some(id), _) => self.entry_for_id(id, delta.index),
            (None, Some(index)) => self.entry_for_index(index),
            (None, None) => self.push_unkeyed(),
        };

        if let Some(id) = delta.id.as_ref()
            && entry.id.as_ref() != Some(id)
        {
            entry.id = Some(id.clone());
        }
        if let Some(index) = delta.index
            && entry.index != Some(index)
        {
            entry.index = Some(index);
        }
        if let Some(function) = delta.function.as_ref() {
            if let Some(name) = function.name.as_ref() {
                entry.name = Some(name.clone());
            }
            if let Some(arguments) = function.arguments.as_ref() {
                entry.arguments.push_str(arguments);
            }
        }
    }

    fn entry_for_id(&mut self, id: &str, index: Option<u32>) -> &mut PartialToolCall {
        if let Some(pos) = self.by_id.get(id).copied() {
            if let Some(index) = index {
                self.by_index.insert(index, pos);
            }
            return &mut self.calls[pos];
        }

        if let Some(index) = index
            && let Some(pos) = self.by_index.remove(&index)
        {
            self.by_id.insert(id.to_string(), pos);
            let entry = &mut self.calls[pos];
            entry.id = Some(id.to_string());
            return entry;
        }

        let pos = self.calls.len();
        self.calls.push(PartialToolCall {
            id: Some(id.to_string()),
            index,
            name: None,
            arguments: String::new(),
        });
        self.by_id.insert(id.to_string(), pos);
        if let Some(index) = index {
            self.by_index.insert(index, pos);
        }
        &mut self.calls[pos]
    }

    fn entry_for_index(&mut self, index: u32) -> &mut PartialToolCall {
        if let Some(pos) = self.by_index.get(&index).copied() {
            return &mut self.calls[pos];
        }

        let pos = self.calls.len();
        self.calls.push(PartialToolCall {
            id: None,
            index: Some(index),
            name: None,
            arguments: String::new(),
        });
        self.by_index.insert(index, pos);
        &mut self.calls[pos]
    }

    fn push_unkeyed(&mut self) -> &mut PartialToolCall {
        let pos = self.calls.len();
        self.calls.push(PartialToolCall::default());
        &mut self.calls[pos]
    }

    fn drain(self) -> Vec<PartialToolCall> {
        self.calls
    }
}

struct ChatCompletionsStreamState {
    response_id: String,
    message: String,
    message_added: bool,
    message_done: bool,
    tool_calls: ToolCallAccumulator,
    tool_calls_emitted: bool,
    token_usage: Option<TokenUsage>,
}

impl ChatCompletionsStreamState {
    fn new() -> Self {
        Self {
            response_id: String::new(),
            message: String::new(),
            message_added: false,
            message_done: false,
            tool_calls: ToolCallAccumulator::default(),
            tool_calls_emitted: false,
            token_usage: None,
        }
    }

    async fn emit_message_added(
        &mut self,
        tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    ) -> bool {
        if self.message_added {
            return true;
        }

        let item = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: Vec::new(),
            end_turn: None,
            phase: None,
        };
        self.message_added = tx_event
            .send(Ok(ResponseEvent::OutputItemAdded(item)))
            .await
            .is_ok();
        self.message_added
    }

    async fn emit_message_done(
        &mut self,
        tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    ) -> bool {
        if self.message_done || self.message.is_empty() {
            return true;
        }

        let item = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: self.message.clone(),
            }],
            end_turn: None,
            phase: None,
        };
        self.message_done = tx_event
            .send(Ok(ResponseEvent::OutputItemDone(item)))
            .await
            .is_ok();
        self.message_done
    }

    async fn emit_tool_calls(
        &mut self,
        tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    ) -> bool {
        if self.tool_calls_emitted {
            return true;
        }
        let calls = std::mem::take(&mut self.tool_calls).drain();
        for (index, call) in calls.into_iter().enumerate() {
            let Some(name) = call.name.as_ref() else {
                debug!("chat tool call missing name; skipping");
                continue;
            };
            let call_id = call
                .id
                .clone()
                .or_else(|| call.index.map(|index| index.to_string()))
                .unwrap_or_else(|| format!("tool_call_{index}"));
            let item = ResponseItem::FunctionCall {
                id: None,
                name: name.clone(),
                arguments: call.arguments,
                call_id,
            };
            if tx_event
                .send(Ok(ResponseEvent::OutputItemDone(item)))
                .await
                .is_err()
            {
                return false;
            }
        }
        self.tool_calls_emitted = true;
        true
    }

    async fn emit_completion(
        &self,
        tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    ) -> bool {
        tx_event
            .send(Ok(ResponseEvent::Completed {
                response_id: self.response_id.clone(),
                token_usage: self.token_usage.clone(),
            }))
            .await
            .is_ok()
    }
}

async fn process_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = ChatCompletionsStreamState::new();

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }

        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(err))) => {
                debug!("SSE Error: {err:#}");
                let _ = tx_event.send(Err(ApiError::Stream(err.to_string()))).await;
                return;
            }
            Ok(None) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "stream closed before completion".into(),
                    )))
                    .await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream("idle timeout waiting for SSE".into())))
                    .await;
                return;
            }
        };

        if sse.data.trim() == "[DONE]" {
            let _ = state.emit_message_done(&tx_event).await;
            let _ = state.emit_tool_calls(&tx_event).await;
            let _ = state.emit_completion(&tx_event).await;
            return;
        }

        trace!("SSE event: {}", &sse.data);
        let event: ChatCompletionsStreamEvent = match serde_json::from_str(&sse.data) {
            Ok(event) => event,
            Err(err) => {
                debug!(
                    "Failed to parse chat completions SSE: {err}, data: {}",
                    &sse.data
                );
                continue;
            }
        };

        if let Some(id) = event.id.as_ref()
            && state.response_id.is_empty()
        {
            state.response_id = id.clone();
        }
        if let Some(usage) = event.usage {
            state.token_usage = Some(usage.into());
        }

        for choice in &event.choices {
            if let Some(content) = choice.delta.content.as_ref()
                && !content.is_empty()
            {
                if !state.emit_message_added(&tx_event).await {
                    return;
                }
                state.message.push_str(content);
                if tx_event
                    .send(Ok(ResponseEvent::OutputTextDelta(content.clone())))
                    .await
                    .is_err()
                {
                    return;
                }
            }

            if let Some(tool_calls) = choice.delta.tool_calls.as_ref() {
                for tool_call in tool_calls {
                    state.tool_calls.apply_delta(tool_call);
                }
            }

            if let Some(finish_reason) = choice.finish_reason.as_ref() {
                match finish_reason.as_str() {
                    "tool_calls" => {
                        if !state.emit_tool_calls(&tx_event).await {
                            return;
                        }
                    }
                    "stop" => {
                        if !state.emit_message_done(&tx_event).await {
                            return;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_client::TransportError;
    use futures::TryStreamExt;
    use pretty_assertions::assert_eq;
    use tokio::sync::mpsc;
    use tokio_util::io::ReaderStream;

    async fn collect_events(chunks: &[&[u8]]) -> Vec<Result<ResponseEvent, ApiError>> {
        let mut reader = Vec::new();
        for chunk in chunks {
            reader.extend_from_slice(chunk);
        }
        let stream = ReaderStream::new(std::io::Cursor::new(reader))
            .map_err(|err| TransportError::Network(err.to_string()));
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(16);
        tokio::spawn(process_sse(Box::pin(stream), tx, idle_timeout(), None));

        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    }

    fn idle_timeout() -> Duration {
        Duration::from_millis(1000)
    }

    #[tokio::test]
    async fn streams_text_and_completes() {
        let sse = concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        let events = collect_events(&[sse.as_bytes()]).await;
        let ok_events: Vec<ResponseEvent> = events.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            ok_events,
            vec![
                ResponseEvent::OutputItemAdded(ResponseItem::Message {
                    id: None,
                    role: "assistant".into(),
                    content: Vec::new(),
                    end_turn: None,
                    phase: None
                }),
                ResponseEvent::OutputTextDelta("hi".to_string()),
                ResponseEvent::OutputItemDone(ResponseItem::Message {
                    id: None,
                    role: "assistant".into(),
                    content: vec![ContentItem::OutputText {
                        text: "hi".to_string()
                    }],
                    end_turn: None,
                    phase: None
                }),
                ResponseEvent::Completed {
                    response_id: "c1".to_string(),
                    token_usage: Some(TokenUsage {
                        input_tokens: 1,
                        cached_input_tokens: 0,
                        output_tokens: 1,
                        reasoning_output_tokens: 0,
                        total_tokens: 2,
                    })
                }
            ]
        );
    }

    #[tokio::test]
    async fn streams_tool_calls() {
        let sse = concat!(
            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"tool\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let events = collect_events(&[sse.as_bytes()]).await;
        let ok_events: Vec<ResponseEvent> = events.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            ok_events,
            vec![
                ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                    id: None,
                    name: "tool".into(),
                    arguments: "{\"a\":1}".into(),
                    call_id: "call-1".into(),
                }),
                ResponseEvent::Completed {
                    response_id: "c2".to_string(),
                    token_usage: None,
                }
            ]
        );
    }

    #[test]
    fn non_stream_response_to_events() {
        let response = ChatCompletionsResponse {
            id: "resp-1".to_string(),
            choices: vec![ChatCompletionsResponseChoice {
                message: ChatCompletionsResponseMessage {
                    role: "assistant".to_string(),
                    content: Some("hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(ChatCompletionsUsage {
                prompt_tokens: 2,
                completion_tokens: 3,
                total_tokens: 5,
            }),
        };

        let events = chat_completions_response_to_events(response);
        assert_eq!(
            events,
            vec![
                ResponseEvent::OutputItemDone(ResponseItem::Message {
                    id: None,
                    role: "assistant".to_string(),
                    content: vec![ContentItem::OutputText {
                        text: "hello".to_string(),
                    }],
                    end_turn: None,
                    phase: None,
                }),
                ResponseEvent::Completed {
                    response_id: "resp-1".to_string(),
                    token_usage: Some(TokenUsage {
                        input_tokens: 2,
                        cached_input_tokens: 0,
                        output_tokens: 3,
                        reasoning_output_tokens: 0,
                        total_tokens: 5,
                    }),
                }
            ]
        );
    }
}
