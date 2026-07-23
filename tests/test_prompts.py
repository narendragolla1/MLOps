import pytest

from omniai.prompts import (
    ChatPromptTemplate,
    FewShotChatPromptTemplate,
    MessagesPlaceholder,
    PromptError,
    PromptTemplate,
)
from omniai.protocol import OmniMessage, Role


def test_prompt_template_variables_and_format():
    template = PromptTemplate("Translate {text} into {language}.")
    assert template.input_variables == {"text", "language"}
    assert template.format(text="hi", language="French") == "Translate hi into French."


def test_missing_variable_raises():
    with pytest.raises(PromptError, match="language"):
        PromptTemplate("To {language}: {text}").format(text="hi")


def test_partial_application():
    template = PromptTemplate("To {language}: {text}").partial(language="German")
    assert template.input_variables == {"text"}
    assert template.format(text="hello") == "To German: hello"


def test_chat_template_roles_and_history():
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a {profession}."),
            MessagesPlaceholder("history"),
            ("user", "{question}"),
        ]
    )
    assert template.input_variables == {"profession", "history", "question"}
    messages = template.format_messages(
        profession="pirate",
        history=[{"role": "user", "content": "ahoy"}, {"role": "assistant", "content": "arr"}],
        question="where's the treasure?",
    )
    assert messages == [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "ahoy"},
        {"role": "assistant", "content": "arr"},
        {"role": "user", "content": "where's the treasure?"},
    ]


def test_placeholder_accepts_omni_messages_and_optional():
    template = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("history", optional=True), ("user", "{q}")]
    )
    assert template.format_messages(q="hi") == [{"role": "user", "content": "hi"}]
    messages = template.format_messages(
        q="hi", history=[OmniMessage(role=Role.ASSISTANT, content="hello")]
    )
    assert messages[0] == {"role": "assistant", "content": "hello"}


def test_missing_placeholder_raises():
    template = ChatPromptTemplate.from_messages([MessagesPlaceholder("history")])
    with pytest.raises(PromptError, match="history"):
        template.format_messages()


def test_few_shot_examples_nest_in_chat_template():
    few_shot = FewShotChatPromptTemplate(
        examples=[
            {"input": "2+2", "output": "4"},
            {"input": "3+3", "output": "6"},
        ]
    )
    template = ChatPromptTemplate.from_messages(
        [("system", "Answer with just the number."), few_shot, ("user", "{question}")]
    )
    messages = template.format_messages(question="5+5")
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1] == {"role": "user", "content": "2+2"}
    assert messages[2] == {"role": "assistant", "content": "4"}
    assert messages[-1] == {"role": "user", "content": "5+5"}
