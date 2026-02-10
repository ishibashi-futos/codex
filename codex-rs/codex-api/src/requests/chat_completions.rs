use crate::error::ApiError;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatCompletionsToolCall {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: ChatCompletionsFunctionCall,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatCompletionsFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatCompletionsMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionsToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionsResponseFormat {
    JsonSchema {
        json_schema: ChatCompletionsJsonSchema,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatCompletionsJsonSchema {
    pub name: String,
    pub schema: Value,
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatCompletionsStreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionsMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ChatCompletionsResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionsStreamOptions>,
}

#[derive(Default)]
pub struct ChatCompletionsRequestBuilder<'a> {
    model: Option<&'a str>,
    messages: Option<&'a [ChatCompletionsMessage]>,
    tools: Option<&'a [Value]>,
    tool_choice: Option<String>,
    parallel_tool_calls: Option<bool>,
    response_format: Option<ChatCompletionsResponseFormat>,
    stream: Option<bool>,
    stream_options: Option<ChatCompletionsStreamOptions>,
}

impl<'a> ChatCompletionsRequestBuilder<'a> {
    pub fn new(model: &'a str, messages: &'a [ChatCompletionsMessage]) -> Self {
        Self {
            model: Some(model),
            messages: Some(messages),
            ..Default::default()
        }
    }

    pub fn tools(mut self, tools: &'a [Value]) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn tool_choice(mut self, tool_choice: Option<String>) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    pub fn parallel_tool_calls(mut self, parallel_tool_calls: Option<bool>) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    pub fn response_format(
        mut self,
        response_format: Option<ChatCompletionsResponseFormat>,
    ) -> Self {
        self.response_format = response_format;
        self
    }

    pub fn stream(mut self, stream: Option<bool>) -> Self {
        self.stream = stream;
        self
    }

    pub fn stream_options(mut self, stream_options: Option<ChatCompletionsStreamOptions>) -> Self {
        self.stream_options = stream_options;
        self
    }

    pub fn build(self) -> Result<ChatCompletionsRequest, ApiError> {
        let model = self
            .model
            .ok_or_else(|| ApiError::Stream("missing model for chat completions request".into()))?;
        let messages = self.messages.ok_or_else(|| {
            ApiError::Stream("missing messages for chat completions request".into())
        })?;

        let tools = self
            .tools
            .map(<[Value]>::to_vec)
            .filter(|tools| !tools.is_empty());
        let tool_choice = if tools.is_some() {
            self.tool_choice
        } else {
            None
        };
        let parallel_tool_calls = if tools.is_some() {
            self.parallel_tool_calls
        } else {
            None
        };
        let stream_options = if self.stream == Some(true) {
            self.stream_options
        } else {
            None
        };

        Ok(ChatCompletionsRequest {
            model: model.to_string(),
            messages: messages.to_vec(),
            tools,
            tool_choice,
            parallel_tool_calls,
            response_format: self.response_format,
            stream: self.stream,
            stream_options,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn serializes_chat_request_with_tools_and_response_format() {
        let messages = vec![ChatCompletionsMessage {
            role: "system".to_string(),
            content: Some(Value::String("hello".to_string())),
            tool_calls: None,
            tool_call_id: None,
        }];

        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"]
                }
            }
        })];

        let response_format = ChatCompletionsResponseFormat::JsonSchema {
            json_schema: ChatCompletionsJsonSchema {
                name: "schema".to_string(),
                schema: json!({"type": "object"}),
                strict: true,
            },
        };

        let request = ChatCompletionsRequestBuilder::new("gpt-test", &messages)
            .tools(&tools)
            .tool_choice(Some("auto".to_string()))
            .parallel_tool_calls(Some(true))
            .response_format(Some(response_format))
            .stream(Some(true))
            .stream_options(Some(ChatCompletionsStreamOptions {
                include_usage: true,
            }))
            .build()
            .expect("request");

        let json = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(json["model"], json!("gpt-test"));
        assert_eq!(json["messages"], json!(messages));
        assert_eq!(json["tools"], json!(tools));
        assert_eq!(json["tool_choice"], json!("auto"));
        assert_eq!(json["parallel_tool_calls"], json!(true));
        assert_eq!(json["response_format"]["type"], json!("json_schema"));
        assert_eq!(
            json["response_format"]["json_schema"]["name"],
            json!("schema")
        );
        assert_eq!(json["stream"], json!(true));
        assert_eq!(json["stream_options"]["include_usage"], json!(true));
    }
}
