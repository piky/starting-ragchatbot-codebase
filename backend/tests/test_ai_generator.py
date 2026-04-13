"""
Tests for AIGenerator (ai_generator.py).
Verifies tool conversion, tool execution, and error handling.
"""
import pytest
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_message_mock(content=None, tool_calls=None):
    # Structure needs to match: response.message.content
    msg = MagicMock()
    response = MagicMock()
    response.message = msg
    msg.content = content
    msg.tool_calls = tool_calls
    return response


def make_tool_call(name, arguments):
    tc = MagicMock()
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# _convert_tools_to_ollama()
# ---------------------------------------------------------------------------

class TestConvertToolsToOllama:
    def test_converts_anthropic_style_to_ollama_format(self):
        gen = AIGenerator(model="test-model")
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            }
        ]

        result = gen._convert_tools_to_ollama(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search_course_content"
        assert result[0]["function"]["description"] == "Search course materials"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_converts_multiple_tools(self):
        gen = AIGenerator(model="test-model")
        tools = [
            {"name": "tool_a", "description": "A", "input_schema": {}},
            {"name": "tool_b", "description": "B", "input_schema": {}},
        ]

        result = gen._convert_tools_to_ollama(tools)

        assert len(result) == 2
        names = {r["function"]["name"] for r in result}
        assert names == {"tool_a", "tool_b"}

    def test_handles_tools_without_description(self):
        gen = AIGenerator(model="test-model")
        tools = [{"name": "tool_no_desc", "input_schema": {}}]

        result = gen._convert_tools_to_ollama(tools)

        assert result[0]["function"]["description"] == ""

    def test_handles_empty_tool_list(self):
        gen = AIGenerator(model="test-model")
        result = gen._convert_tools_to_ollama([])
        assert result == []


# ---------------------------------------------------------------------------
# generate_response() — direct responses (no tools)
# ---------------------------------------------------------------------------

class TestGenerateResponseDirect:
    def test_returns_content_from_ollama_response(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(
                content="Python functions use the `def` keyword."
            )
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response("What is a Python function?")

            assert result == "Python functions use the `def` keyword."
            MockClient.return_value.chat.assert_called_once()

    def test_passes_system_prompt_and_user_query(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content="answer")
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response("What is 2+2?")

            call_kwargs = MockClient.return_value.chat.call_args.kwargs
            messages = call_kwargs["messages"]

            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "What is 2+2?"

    def test_includes_conversation_history_in_system_prompt(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content="answer")
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response(
                "What about inheritance?",
                conversation_history="User: What is a class?\nAssistant: A class is a blueprint.",
            )

            call_kwargs = MockClient.return_value.chat.call_args.kwargs
            system_content = call_kwargs["messages"][0]["content"]
            assert "User: What is a class?" in system_content

    def test_includes_tools_in_api_call_when_provided(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content="answer")
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            tools = [{"name": "search_course_content", "description": "", "input_schema": {}}]

            gen.generate_response("Query", tools=tools)

            call_kwargs = MockClient.return_value.chat.call_args.kwargs
            assert "tools" in call_kwargs
            assert call_kwargs["tools"][0]["function"]["name"] == "search_course_content"

    def test_does_not_include_tools_when_not_provided(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content="answer")
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response("Query")

            call_kwargs = MockClient.return_value.chat.call_args.kwargs
            assert "tools" not in call_kwargs

    def test_returns_empty_string_when_content_is_none(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content=None)
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response("Query")

            assert result == ""


# ---------------------------------------------------------------------------
# generate_response() — tool execution path
# ---------------------------------------------------------------------------

class TestGenerateResponseWithToolExecution:
    def test_calls_tool_manager_when_tool_calls_present(self):
        with patch("ai_generator.Client") as MockClient:
            tool_call = make_tool_call(
                "search_course_content",
                {"query": "Python functions", "course_name": "Python", "lesson_number": None},
            )
            mock_response_with_tool = make_message_mock(
                content=None, tool_calls=[tool_call]
            )

            tool_result = "Python functions are defined using `def`."
            final_response = make_message_mock(
                content="Here is the answer based on search results."
            )

            MockClient.return_value.chat.side_effect = [
                mock_response_with_tool,
                final_response,
            ]

            # Build a mock tool manager
            mock_tm = MagicMock()
            mock_tm.execute_tool.return_value = tool_result

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response(
                "What are Python functions?",
                tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            mock_tm.execute_tool.assert_called_once_with(
                "search_course_content",
                query="Python functions",
                course_name="Python",
                lesson_number=None,
            )
            assert "based on search results" in result

    def test_injects_tool_result_into_messages(self):
        with patch("ai_generator.Client") as MockClient:
            tool_call = make_tool_call("search_course_content", {"query": "test"})
            mock_response_with_tool = make_message_mock(content=None, tool_calls=[tool_call])
            final_response = make_message_mock(content="Final answer.")

            MockClient.return_value.chat.side_effect = [mock_response_with_tool, final_response]

            mock_tm = MagicMock()
            mock_tm.execute_tool.return_value = "Search found: test content."

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response("test", tool_manager=mock_tm)

            # Second call to chat should include the tool result message
            second_call_kwargs = MockClient.return_value.chat.call_args_list[1].kwargs
            all_messages = second_call_kwargs["messages"]

            tool_msg = next(
                (m for m in all_messages if m.get("role") == "tool"),
                None,
            )
            assert tool_msg is not None, "No tool message found in second API call"
            assert tool_msg["content"] == "Search found: test content."
            assert tool_msg["tool_name"] == "search_course_content"

    def test_handles_multiple_tool_calls_in_single_response(self):
        with patch("ai_generator.Client") as MockClient:
            tool_call_a = make_tool_call("search_course_content", {"query": "a"})
            tool_call_b = make_tool_call("search_course_content", {"query": "b"})
            mock_response = make_message_mock(content=None, tool_calls=[tool_call_a, tool_call_b])
            final_response = make_message_mock(content="Combined answer.")

            MockClient.return_value.chat.side_effect = [mock_response, final_response]

            mock_tm = MagicMock()
            mock_tm.execute_tool.side_effect = ["Result A", "Result B"]

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response("combined query", tool_manager=mock_tm)

            assert mock_tm.execute_tool.call_count == 2

    def test_passes_correct_model_and_options(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content="answer")
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="my-model", host="http://custom-host:9999")
            gen.generate_response("test")

            call_kwargs = MockClient.return_value.chat.call_args.kwargs
            assert call_kwargs["model"] == "my-model"
            assert call_kwargs["options"]["temperature"] == 0
            assert call_kwargs["options"]["num_predict"] == 800


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_raises_runtime_error_on_request_error(self):
        from ollama import RequestError

        with patch("ai_generator.Client") as MockClient:
            MockClient.return_value.chat.side_effect = RequestError("Connection refused")

            gen = AIGenerator(model="test-model", host="http://localhost:11434")

            with pytest.raises(RuntimeError) as exc_info:
                gen.generate_response("test query")

            assert "Ollama client error" in str(exc_info.value)

    def test_raises_runtime_error_on_response_error(self):
        from ollama import ResponseError

        with patch("ai_generator.Client") as MockClient:
            MockClient.return_value.chat.side_effect = ResponseError("Model not found")

            gen = AIGenerator(model="test-model", host="http://localhost:11434")

            with pytest.raises(RuntimeError) as exc_info:
                gen.generate_response("test query")

            assert "Ollama client error" in str(exc_info.value)

    def test_raises_runtime_error_on_generic_exception(self):
        with patch("ai_generator.Client") as MockClient:
            MockClient.return_value.chat.side_effect = Exception("Unexpected")

            gen = AIGenerator(model="test-model", host="http://localhost:11434")

            with pytest.raises(RuntimeError):
                gen.generate_response("test query")


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT content
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_mentions_search_tool(self):
        gen = AIGenerator(model="test-model")
        # The system prompt is a class attribute
        assert "search" in gen.SYSTEM_PROMPT.lower() or "Search" in gen.SYSTEM_PROMPT

    def test_system_prompt_instructs_against_meta_commentary(self):
        gen = AIGenerator(model="test-model")
        assert "no meta" in gen.SYSTEM_PROMPT.lower() or "meta" in gen.SYSTEM_PROMPT.lower() or "reasoning" in gen.SYSTEM_PROMPT.lower() or "commentary" in gen.SYSTEM_PROMPT.lower() or "without" in gen.SYSTEM_PROMPT.lower()

    def test_base_options_have_temperature_zero(self):
        gen = AIGenerator(model="test-model")
        assert gen.base_options["temperature"] == 0

    def test_base_options_limit_token_predict(self):
        gen = AIGenerator(model="test-model")
        assert gen.base_options["num_predict"] == 800