"""Risk annotations advertised for tools with side effects (#1248)."""

import pytest

import gmail.gmail_tools  # noqa: F401
from core.server import server


@pytest.mark.asyncio
async def test_send_gmail_message_is_destructive():
    tool = await server.get_tool("send_gmail_message")
    assert tool.annotations.read_only_hint is False
    assert tool.annotations.destructive_hint is True
