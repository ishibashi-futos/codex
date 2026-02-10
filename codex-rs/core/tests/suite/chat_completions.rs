use std::sync::Arc;
use std::sync::Mutex;

use codex_core::ContentItem;
use codex_core::ModelClient;
use codex_core::ModelProviderInfo;
use codex_core::Prompt;
use codex_core::ResponseEvent;
use codex_core::ResponseItem;
use codex_core::WireApi;
use codex_core::models_manager::manager::ModelsManager;
use codex_otel::OtelManager;
use codex_otel::TelemetryAuthMode;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::SessionSource;
use codex_protocol::user_input::UserInput;
use core_test_support::load_default_config_for_test;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event_match;
use futures::StreamExt;
use pretty_assertions::assert_eq;
use serde_json::Value;
use tempfile::TempDir;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

enum ChatSseEvent {
    Data(Value),
    Done,
}

fn chat_sse(events: Vec<ChatSseEvent>) -> String {
    let mut body = String::new();
    for event in events {
        match event {
            ChatSseEvent::Data(value) => {
                body.push_str("data: ");
                body.push_str(&value.to_string());
                body.push_str("\n\n");
            }
            ChatSseEvent::Done => {
                body.push_str("data: [DONE]\n\n");
            }
        }
    }
    body
}

#[derive(Clone)]
struct ChatRequestMock {
    requests: Arc<Mutex<Vec<Request>>>,
    response: ResponseTemplate,
}

impl ChatRequestMock {
    fn new(response: ResponseTemplate) -> (Self, Arc<Mutex<Vec<Request>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let mock = Self {
            requests: Arc::clone(&requests),
            response,
        };
        (mock, requests)
    }

    fn single_request_body(&self) -> Value {
        let requests = self
            .requests
            .lock()
            .unwrap_or_else(|err| panic!("request lock poisoned: {err}"));
        assert_eq!(requests.len(), 1);
        serde_json::from_slice(&requests[0].body)
            .unwrap_or_else(|err| panic!("parse request body: {err}"))
    }

    fn request_count(&self) -> usize {
        self.requests
            .lock()
            .unwrap_or_else(|err| panic!("request lock poisoned: {err}"))
            .len()
    }
}

impl Respond for ChatRequestMock {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        self.requests
            .lock()
            .unwrap_or_else(|err| panic!("request lock poisoned: {err}"))
            .push(req.clone());
        self.response.clone()
    }
}

async fn mount_chat_once(server: &MockServer, response: ResponseTemplate) -> ChatRequestMock {
    let (mock, _requests) = ChatRequestMock::new(response);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(mock.clone())
        .up_to_n_times(1)
        .mount(server)
        .await;
    mock
}

async fn build_client_harness(
    provider: ModelProviderInfo,
) -> (
    ModelClient,
    Prompt,
    OtelManager,
    ReasoningSummary,
    ModelInfo,
) {
    let codex_home = TempDir::new().unwrap_or_else(|err| panic!("temp dir: {err}"));
    let mut config = load_default_config_for_test(&codex_home).await;
    config.model_provider_id = provider.name.clone();
    config.model_provider = provider.clone();
    let model = ModelsManager::get_model_offline(config.model.as_deref());
    config.model = Some(model.clone());
    let config = Arc::new(config);
    let model_info = ModelsManager::construct_model_info_offline(model.as_str(), &config);
    let conversation_id = ThreadId::new();
    let session_source = SessionSource::Cli;
    let otel_manager = OtelManager::new(
        conversation_id,
        model.as_str(),
        model_info.slug.as_str(),
        None,
        None,
        Some(TelemetryAuthMode::ApiKey),
        "test_originator".to_string(),
        false,
        "test".to_string(),
        session_source.clone(),
    );
    let client = ModelClient::new(
        None,
        conversation_id,
        provider,
        session_source,
        config.model_verbosity,
        false,
        false,
        false,
        false,
        None,
    );
    let mut prompt = Prompt::default();
    prompt.input = vec![ResponseItem::Message {
        id: None,
        role: "user".into(),
        content: vec![ContentItem::InputText {
            text: "hello".into(),
        }],
        end_turn: None,
        phase: None,
    }];
    (
        client,
        prompt,
        otel_manager,
        config.model_reasoning_summary,
        model_info,
    )
}

fn chat_provider(base_url: String, supports_streaming: bool) -> ModelProviderInfo {
    ModelProviderInfo {
        name: "mock-chat".into(),
        base_url: Some(base_url),
        env_key: None,
        env_key_instructions: None,
        experimental_bearer_token: None,
        wire_api: WireApi::ChatCompletions,
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: Some(0),
        stream_max_retries: Some(0),
        stream_idle_timeout_ms: Some(5_000),
        requires_openai_auth: false,
        supports_websockets: false,
        supports_streaming,
        unsupported_params: Vec::new(),
        supports_response_format: true,
        supports_parallel_tool_calls: true,
    }
}

#[tokio::test]
async fn chat_stream_emits_function_call() {
    core_test_support::skip_if_no_network!();

    let server = start_mock_server().await;
    let sse_body = chat_sse(vec![
        ChatSseEvent::Data(serde_json::json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{\"a\":"}
                    }]
                }
            }]
        })),
        ChatSseEvent::Data(serde_json::json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        })),
        ChatSseEvent::Done,
    ]);

    let response = ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(sse_body, "text/event-stream");
    mount_chat_once(&server, response).await;

    let provider = chat_provider(format!("{}/v1", server.uri()), true);
    let (client, prompt, otel_manager, summary, model_info) = build_client_harness(provider).await;
    let mut session = client.new_session();

    let mut stream = session
        .stream(&prompt, &model_info, &otel_manager, None, summary, None)
        .await
        .expect("stream failed");

    let mut seen = None;
    while let Some(event) = stream.next().await {
        if let Ok(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        })) = event
        {
            seen = Some((name, arguments, call_id));
            break;
        }
    }

    assert_eq!(
        seen,
        Some((
            "tool".to_string(),
            "{\"a\":1}".to_string(),
            "call-1".to_string()
        ))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_emits_output_text_delta_to_ui() -> anyhow::Result<()> {
    core_test_support::skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let sse_body = chat_sse(vec![
        ChatSseEvent::Data(serde_json::json!({
            "id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        })),
        ChatSseEvent::Data(serde_json::json!({
            "id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {"content": "streamed response"}}]
        })),
        ChatSseEvent::Data(serde_json::json!({
            "id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        })),
        ChatSseEvent::Done,
    ]);

    let response = ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(sse_body, "text/event-stream");
    mount_chat_once(&server, response).await;

    let TestCodex { codex, .. } = test_codex()
        .with_config(|config| {
            config.model_provider.wire_api = WireApi::ChatCompletions;
        })
        .build(&server)
        .await?;

    codex
        .submit(codex_core::protocol::Op::UserInput {
            items: vec![UserInput::Text {
                text: "please stream".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
        })
        .await?;

    let delta = wait_for_event_match(&codex, |ev| match ev {
        EventMsg::AgentMessageContentDelta(event) => Some(event.delta.clone()),
        _ => None,
    })
    .await;

    assert_eq!(delta, "streamed response");
    Ok(())
}

#[tokio::test]
async fn supports_streaming_false_uses_non_stream() {
    core_test_support::skip_if_no_network!();

    let server = start_mock_server().await;
    let response = ResponseTemplate::new(200)
        .insert_header("content-type", "application/json")
        .set_body_json(serde_json::json!({
            "id": "chatcmpl-3",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "non stream"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));
    let recorder = mount_chat_once(&server, response).await;

    let provider = chat_provider(format!("{}/v1", server.uri()), false);
    let (client, prompt, otel_manager, summary, model_info) = build_client_harness(provider).await;
    let mut session = client.new_session();

    let mut stream = session
        .stream(&prompt, &model_info, &otel_manager, None, summary, None)
        .await
        .expect("stream failed");

    while let Some(event) = stream.next().await {
        if matches!(event, Ok(ResponseEvent::Completed { .. })) {
            break;
        }
    }

    let body = recorder.single_request_body();
    assert_ne!(body.get("stream"), Some(&Value::Bool(true)));
}

#[tokio::test]
async fn partial_output_does_not_fallback_to_non_stream() {
    core_test_support::skip_if_no_network!();

    let server = start_mock_server().await;
    let sse_body = chat_sse(vec![ChatSseEvent::Data(serde_json::json!({
        "id": "chatcmpl-4",
        "choices": [{"index": 0, "delta": {"content": "partial"}}]
    }))]);
    let response = ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(sse_body, "text/event-stream");
    let recorder = mount_chat_once(&server, response).await;

    let provider = chat_provider(format!("{}/v1", server.uri()), true);
    let (client, prompt, otel_manager, summary, model_info) = build_client_harness(provider).await;
    let mut session = client.new_session();

    let mut stream = session
        .stream(&prompt, &model_info, &otel_manager, None, summary, None)
        .await
        .expect("stream failed");

    let mut saw_error = false;
    while let Some(event) = stream.next().await {
        if event.is_err() {
            saw_error = true;
            break;
        }
    }

    assert!(saw_error);
    assert_eq!(recorder.request_count(), 1);
}
