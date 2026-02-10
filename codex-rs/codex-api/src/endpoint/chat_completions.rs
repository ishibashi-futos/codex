use crate::auth::AuthProvider;
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::ChatCompletionsRequest;
use crate::sse::chat_completions_response_to_events;
use crate::sse::parse_chat_completions_response;
use crate::sse::spawn_chat_completions_stream;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use std::sync::Arc;
use tracing::instrument;

pub struct ChatCompletionsClient<T: HttpTransport, A: AuthProvider> {
    session: EndpointSession<T, A>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

impl<T: HttpTransport, A: AuthProvider> ChatCompletionsClient<T, A> {
    pub fn new(transport: T, provider: Provider, auth: A) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    fn path() -> &'static str {
        "chat/completions"
    }

    #[instrument(level = "trace", skip_all, err)]
    pub async fn stream_request(
        &self,
        request: ChatCompletionsRequest,
        extra_headers: HeaderMap,
    ) -> Result<ResponseStream, ApiError> {
        let body = serde_json::to_value(&request)
            .map_err(|err| ApiError::Stream(format!("failed to encode chat request: {err}")))?;
        let stream_response = self
            .session
            .stream_with(
                Method::POST,
                Self::path(),
                extra_headers,
                Some(body),
                |req| {
                    req.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                },
            )
            .await?;

        Ok(spawn_chat_completions_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
        ))
    }

    #[instrument(level = "trace", skip_all, err)]
    pub async fn request_events(
        &self,
        request: ChatCompletionsRequest,
        extra_headers: HeaderMap,
    ) -> Result<Vec<ResponseEvent>, ApiError> {
        let body = serde_json::to_value(&request)
            .map_err(|err| ApiError::Stream(format!("failed to encode chat request: {err}")))?;
        let response = self
            .session
            .execute(Method::POST, Self::path(), extra_headers, Some(body))
            .await?;

        let parsed = parse_chat_completions_response(&response.body)?;
        Ok(chat_completions_response_to_events(parsed))
    }
}
