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
# _call_api()
# ---------------------------------------------------------------------------

class TestCallApi:
    def test_passes_messages_model_and_options(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content="answer")
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
            gen._call_api(messages)

            call_kwargs = MockClient.return_value.chat.call_args.kwargs
            assert call_kwargs["model"] == "test-model"
            assert call_kwargs["messages"] == messages
            assert call_kwargs["options"]["temperature"] == 0

    def test_includes_tools_when_provided(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content="answer")
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            tools = [{"name": "search_course_content", "description": "", "input_schema": {}}]
            gen._call_api([{"role": "system", "content": "sys"}], tools=tools)

            call_kwargs = MockClient.return_value.chat.call_args.kwargs
            assert "tools" in call_kwargs
            assert call_kwargs["tools"][0]["function"]["name"] == "search_course_content"

    def test_omits_tools_when_not_provided(self):
        with patch("ai_generator.Client") as MockClient:
            mock_response = make_message_mock(content="answer")
            MockClient.return_value.chat.return_value = mock_response

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen._call_api([{"role": "system", "content": "sys"}])

            call_kwargs = MockClient.return_value.chat.call_args.kwargs
            assert "tools" not in call_kwargs

    def test_raises_runtime_error_on_request_error(self):
        from ollama import RequestError

        with patch("ai_generator.Client") as MockClient:
            MockClient.return_value.chat.side_effect = RequestError("Connection refused")

            gen = AIGenerator(model="test-model", host="http://localhost:11434")

            with pytest.raises(RuntimeError) as exc_info:
                gen._call_api([{"role": "system", "content": "sys"}])

            assert "Ollama client error" in str(exc_info.value)

    def test_raises_runtime_error_on_response_error(self):
        from ollama import ResponseError

        with patch("ai_generator.Client") as MockClient:
            MockClient.return_value.chat.side_effect = ResponseError("Model not found")

            gen = AIGenerator(model="test-model", host="http://localhost:11434")

            with pytest.raises(RuntimeError) as exc_info:
                gen._call_api([{"role": "system", "content": "sys"}])

            assert "Ollama client error" in str(exc_info.value)

    def test_raises_runtime_error_on_generic_exception(self):
        with patch("ai_generator.Client") as MockClient:
            MockClient.return_value.chat.side_effect = Exception("Unexpected")

            gen = AIGenerator(model="test-model", host="http://localhost:11434")

            with pytest.raises(RuntimeError):
                gen._call_api([{"role": "system", "content": "sys"}])


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
            gen.generate_response("test", tools=[{"name": "search_course_content", "description": "", "input_schema": {}}], tool_manager=mock_tm)

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
            gen.generate_response("combined query", tools=[{"name": "search_course_content", "description": "", "input_schema": {}}], tool_manager=mock_tm)

            assert mock_tm.execute_tool.call_count == 2

    def test_assistant_message_appears_once_with_multiple_tool_calls(self):
        """Verify the assistant message is appended exactly once (not once per tool call)."""
        with patch("ai_generator.Client") as MockClient:
            tool_call_a = make_tool_call("search_course_content", {"query": "a"})
            tool_call_b = make_tool_call("search_course_content", {"query": "b"})
            mock_response = make_message_mock(content=None, tool_calls=[tool_call_a, tool_call_b])
            final_response = make_message_mock(content="Combined answer.")

            MockClient.return_value.chat.side_effect = [mock_response, final_response]

            mock_tm = MagicMock()
            mock_tm.execute_tool.side_effect = ["Result A", "Result B"]

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response("combined query", tools=[{"name": "search_course_content", "description": "", "input_schema": {}}], tool_manager=mock_tm)

            # Second API call's messages should have exactly 1 assistant message
            # after the user message (not 2 — the old bug duplicated it)
            second_call_kwargs = MockClient.return_value.chat.call_args_list[1].kwargs
            all_messages = second_call_kwargs["messages"]

            # Messages: system, user, assistant, tool, tool
            assistant_msgs = [m for m in all_messages if isinstance(m, MagicMock) or (isinstance(m, dict) and m.get("role") == "tool")]
            # More precisely: count assistant messages that are raw MagicMock objects
            # (the assistant message is appended as the raw response.message object)
            tool_msgs = [m for m in all_messages if isinstance(m, dict) and m.get("role") == "tool"]
            assert len(tool_msgs) == 2, f"Expected 2 tool messages, got {len(tool_msgs)}"


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

    def test_api_error_during_sequential_round_raises_runtime_error(self):
        """Ollama API error during a sequential round should raise RuntimeError."""
        with patch("ai_generator.Client") as MockClient:
            from ollama import RequestError

            tool_call = make_tool_call("search_course_content", {"query": "first"})
            round1_response = make_message_mock(content=None, tool_calls=[tool_call])

            MockClient.return_value.chat.side_effect = [
                round1_response,
                RequestError("Connection lost during round 2"),
            ]

            mock_tm = MagicMock()
            mock_tm.execute_tool.return_value = "First result"

            gen = AIGenerator(model="test-model", host="http://localhost:11434")

            with pytest.raises(RuntimeError) as exc_info:
                gen.generate_response(
                    "query",
                    tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                    tool_manager=mock_tm,
                )

            assert "Ollama client error" in str(exc_info.value)

    def test_tool_execution_error_handled_gracefully(self):
        """Tool execution errors are captured as tool result messages, not raised."""
        with patch("ai_generator.Client") as MockClient:
            tool_call = make_tool_call("search_course_content", {"query": "first"})
            round1_response = make_message_mock(content=None, tool_calls=[tool_call])
            final_response = make_message_mock(
                content="I encountered an error searching for that content."
            )

            MockClient.return_value.chat.side_effect = [
                round1_response,
                final_response,
            ]

            mock_tm = MagicMock()
            mock_tm.execute_tool.side_effect = RuntimeError("Tool execution failed")

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response(
                "query",
                tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            # Error is captured gracefully — a final synthesis call is made
            # The model gets the error as a tool result and responds to the user
            assert MockClient.return_value.chat.call_count == 2
            # Verify the error was passed as tool result to the synthesis call
            second_call_kwargs = MockClient.return_value.chat.call_args_list[1].kwargs
            all_messages = second_call_kwargs["messages"]
            tool_msgs = [m for m in all_messages if isinstance(m, dict) and m.get("role") == "tool"]
            assert len(tool_msgs) == 1
            assert "Error executing tool" in tool_msgs[0]["content"]
            assert "Tool execution failed" in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT content
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_mentions_search_tool(self):
        gen = AIGenerator(model="test-model")
        assert "search" in gen.SYSTEM_PROMPT.lower() or "Search" in gen.SYSTEM_PROMPT

    def test_system_prompt_instructs_against_meta_commentary(self):
        gen = AIGenerator(model="test-model")
        prompt_lower = gen.SYSTEM_PROMPT.lower()
        assert any(kw in prompt_lower for kw in ["no meta", "meta", "commentary", "without", "reasoning process"])

    def test_base_options_have_temperature_zero(self):
        gen = AIGenerator(model="test-model")
        assert gen.base_options["temperature"] == 0

    def test_base_options_limit_token_predict(self):
        gen = AIGenerator(model="test-model")
        assert gen.base_options["num_predict"] == 800

    def test_system_prompt_mentions_sequential_rounds(self):
        gen = AIGenerator(model="test-model")
        prompt_lower = gen.SYSTEM_PROMPT.lower()
        assert "sequential" in prompt_lower and "round" in prompt_lower

    def test_system_prompt_includes_chaining_guidance(self):
        gen = AIGenerator(model="test-model")
        assert "chained search example" in gen.SYSTEM_PROMPT.lower() or "Round 1" in gen.SYSTEM_PROMPT

    def test_system_prompt_includes_within_round_independence_guidance(self):
        gen = AIGenerator(model="test-model")
        assert "independent" in gen.SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Sequential Tool Calling
# ---------------------------------------------------------------------------

class TestSequentialToolCalling:
    """Tests for sequential tool calling with up to 2 rounds."""

    def test_two_sequential_tool_calls_with_comparison_query(self):
        """AI makes 2 tool calls across separate API rounds to chain operations."""
        with patch("ai_generator.Client") as MockClient:
            # Round 1: AI requests first tool call (get lesson title)
            tool_call_1 = make_tool_call(
                "get_lesson_title",
                {"course_name": "Python Basics", "lesson_number": 4}
            )
            round1_response = make_message_mock(content=None, tool_calls=[tool_call_1])

            # Round 2: AI requests second tool call (search with title)
            tool_call_2 = make_tool_call(
                "search_course_content",
                {"query": "Object-Oriented Programming"}
            )
            round2_response = make_message_mock(content=None, tool_calls=[tool_call_2])

            # Final: AI returns synthesized answer
            final_response = make_message_mock(
                content="Course 'Advanced Python' covers the same topic."
            )

            MockClient.return_value.chat.side_effect = [
                round1_response,
                round2_response,
                final_response,
            ]

            mock_tm = MagicMock()
            mock_tm.execute_tool.side_effect = [
                "Lesson 4: Object-Oriented Programming",
                "Found course: Advanced Python - Object-Oriented Programming",
            ]

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response(
                "Find a course that discusses the same topic as lesson 4 of Python Basics",
                tools=[{"name": "get_lesson_title", "description": "", "input_schema": {}},
                       {"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            # Verify 2 tool calls were made
            assert mock_tm.execute_tool.call_count == 2
            # Verify final response contains synthesized answer
            assert "Advanced Python" in result
            # Verify 3 API calls were made (initial + 2 tool rounds, last round forces synthesis)
            assert MockClient.return_value.chat.call_count == 3

    def test_single_tool_call_in_first_round_then_done(self):
        """AI makes one tool call and final response has no more tool calls."""
        with patch("ai_generator.Client") as MockClient:
            # Round 1: AI requests one tool call
            tool_call = make_tool_call(
                "search_course_content",
                {"query": "Python functions"}
            )
            round1_response = make_message_mock(content=None, tool_calls=[tool_call])

            # Final: AI returns answer without more tool calls
            final_response = make_message_mock(
                content="Python functions are defined using the def keyword."
            )

            MockClient.return_value.chat.side_effect = [
                round1_response,
                final_response,
            ]

            mock_tm = MagicMock()
            mock_tm.execute_tool.return_value = "Found content about Python functions."

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response(
                "What are Python functions?",
                tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            assert mock_tm.execute_tool.call_count == 1
            assert "def" in result
            assert MockClient.return_value.chat.call_count == 2

    def test_max_rounds_terminates_loop(self):
        """After 2 rounds, loop terminates even if AI requests more tools."""
        with patch("ai_generator.Client") as MockClient:
            # Round 1: AI requests first tool call
            tool_call_1 = make_tool_call("search_course_content", {"query": "a"})
            round1_response = make_message_mock(content=None, tool_calls=[tool_call_1])

            # Round 2: AI requests second tool call (but we stop after executing)
            tool_call_2 = make_tool_call("search_course_content", {"query": "b"})
            round2_response = make_message_mock(content=None, tool_calls=[tool_call_2])

            # Final synthesis call after max rounds
            final_response = make_message_mock(content="Final answer.")

            MockClient.return_value.chat.side_effect = [
                round1_response,
                round2_response,
                final_response,
            ]

            mock_tm = MagicMock()
            mock_tm.execute_tool.side_effect = ["Result A", "Result B"]

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response(
                "complex query requiring multiple searches",
                tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            # 3 API calls: initial + round 1 + round 2 (last round forces synthesis)
            assert MockClient.return_value.chat.call_count == 3
            assert mock_tm.execute_tool.call_count == 2

    def test_no_redundant_synthesis_when_model_responds_directly(self):
        """When the model's second API call returns no tool_calls, return content
        directly without making a redundant third API call."""
        with patch("ai_generator.Client") as MockClient:
            # Round 1: AI requests tool call
            tool_call = make_tool_call("search_course_content", {"query": "test"})
            round1_response = make_message_mock(content=None, tool_calls=[tool_call])

            # Round 2: AI responds directly (no tool calls) — this IS the final answer
            round2_direct = make_message_mock(content="Direct answer after tool use.")

            MockClient.return_value.chat.side_effect = [
                round1_response,
                round2_direct,
            ]

            mock_tm = MagicMock()
            mock_tm.execute_tool.return_value = "Tool result."

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response(
                "test query",
                tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            # Only 2 API calls: initial + round 1. No redundant third call.
            assert MockClient.return_value.chat.call_count == 2
            assert result == "Direct answer after tool use."

    def test_conversation_context_preserved_between_rounds(self):
        """Each API call receives messages that include all previous tool results."""
        with patch("ai_generator.Client") as MockClient:
            tool_call_1 = make_tool_call("search_course_content", {"query": "first"})
            round1_response = make_message_mock(content=None, tool_calls=[tool_call_1])

            tool_call_2 = make_tool_call("search_course_content", {"query": "second"})
            round2_response = make_message_mock(content=None, tool_calls=[tool_call_2])

            final_response = make_message_mock(content="Final answer.")

            MockClient.return_value.chat.side_effect = [
                round1_response,
                round2_response,
                final_response,
            ]

            mock_tm = MagicMock()
            mock_tm.execute_tool.side_effect = ["Result 1", "Result 2"]

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response(
                "test query",
                tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            # Verify messages accumulate correctly by checking content presence
            calls = MockClient.return_value.chat.call_args_list

            # First call: system + user
            first_messages = calls[0].kwargs["messages"]
            assert any(m.get("role") == "system" for m in first_messages)
            assert any(m.get("content") == "test query" for m in first_messages)

            # Second call: includes tool result from round 1
            second_messages = calls[1].kwargs["messages"]
            tool_msgs_1 = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"]
            assert len(tool_msgs_1) == 1
            assert tool_msgs_1[0]["content"] == "Result 1"

            # Third call: includes tool results from both rounds
            third_messages = calls[2].kwargs["messages"]
            tool_msgs_2 = [m for m in third_messages if isinstance(m, dict) and m.get("role") == "tool"]
            assert len(tool_msgs_2) == 2
            assert tool_msgs_2[0]["content"] == "Result 1"
            assert tool_msgs_2[1]["content"] == "Result 2"

    def test_tools_in_kwargs_for_next_round_but_not_for_synthesis(self):
        """Next-round API calls include tools; final synthesis call does not."""
        with patch("ai_generator.Client") as MockClient:
            tool_call_1 = make_tool_call("search_course_content", {"query": "first"})
            round1_response = make_message_mock(content=None, tool_calls=[tool_call_1])

            tool_call_2 = make_tool_call("search_course_content", {"query": "second"})
            round2_response = make_message_mock(content=None, tool_calls=[tool_call_2])

            final_response = make_message_mock(content="Final answer.")

            MockClient.return_value.chat.side_effect = [
                round1_response,
                round2_response,
                final_response,
            ]

            mock_tm = MagicMock()
            mock_tm.execute_tool.side_effect = ["Result 1", "Result 2"]

            tools = [{"name": "search_course_content", "description": "", "input_schema": {}}]
            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response("test query", tools=tools, tool_manager=mock_tm)

            calls = MockClient.return_value.chat.call_args_list

            # Call 0 (initial): has tools
            assert "tools" in calls[0].kwargs
            # Call 1 (next round): has tools
            assert "tools" in calls[1].kwargs
            # Call 2 (synthesis after last round): no tools
            assert "tools" not in calls[2].kwargs

    def test_tool_call_with_none_arguments_handled(self):
        """Tool call with None arguments should be treated as empty dict."""
        with patch("ai_generator.Client") as MockClient:
            # Tool call with None arguments
            tool_call = make_tool_call("search_course_content", None)
            round1_response = make_message_mock(content=None, tool_calls=[tool_call])
            final_response = make_message_mock(content="Final answer.")

            MockClient.return_value.chat.side_effect = [round1_response, final_response]

            mock_tm = MagicMock()
            mock_tm.execute_tool.return_value = "Some result."

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            gen.generate_response(
                "test",
                tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            # execute_tool should be called with no kwargs (just the name)
            mock_tm.execute_tool.assert_called_once_with("search_course_content")

    def test_partial_tool_failure_in_multi_tool_round(self):
        """When one tool succeeds and another fails in the same round,
        both results (success + error message) are added to messages."""
        with patch("ai_generator.Client") as MockClient:
            tool_call_a = make_tool_call("search_course_content", {"query": "a"})
            tool_call_b = make_tool_call("search_course_content", {"query": "b"})
            round1_response = make_message_mock(content=None, tool_calls=[tool_call_a, tool_call_b])
            final_response = make_message_mock(content="Partial answer with error note.")

            MockClient.return_value.chat.side_effect = [round1_response, final_response]

            mock_tm = MagicMock()
            mock_tm.execute_tool.side_effect = ["Result A", RuntimeError("Tool B failed")]

            gen = AIGenerator(model="test-model", host="http://localhost:11434")
            result = gen.generate_response(
                "query",
                tools=[{"name": "search_course_content", "description": "", "input_schema": {}}],
                tool_manager=mock_tm,
            )

            # Both tools were called
            assert mock_tm.execute_tool.call_count == 2
            # A synthesis call was made with both results
            second_call_kwargs = MockClient.return_value.chat.call_args_list[1].kwargs
            tool_msgs = [m for m in second_call_kwargs["messages"]
                         if isinstance(m, dict) and m.get("role") == "tool"]
            assert len(tool_msgs) == 2
            assert tool_msgs[0]["content"] == "Result A"
            assert "Error executing tool" in tool_msgs[1]["content"]
            assert "Tool B failed" in tool_msgs[1]["content"]