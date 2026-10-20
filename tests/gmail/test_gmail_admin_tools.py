import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.utils import UserInputError
from gmail import gmail_admin_tools as admin

U = "user@example.com"


def _unwrap(tool):
    fn = tool.fn if hasattr(tool, "fn") else tool
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


# --- filters -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_trash_messages_and_threads():
    svc = MagicMock()
    out = await _unwrap(admin.manage_gmail_trash)(
        service=svc,
        user_google_email=U,
        action="trash",
        message_ids=["m1"],
        thread_ids=["t1"],
    )
    svc.users().messages().trash.assert_called_with(userId="me", id="m1")
    svc.users().threads().trash.assert_called_with(userId="me", id="t1")
    assert "Trashed 2 item(s)." in out


@pytest.mark.asyncio
async def test_untrash_uses_untrash():
    svc = MagicMock()
    await _unwrap(admin.manage_gmail_trash)(
        service=svc, user_google_email=U, action="untrash", message_ids=["m1"]
    )
    svc.users().messages().untrash.assert_called_with(userId="me", id="m1")


def test_permanent_delete_tool_absent_without_opt_in():
    import subprocess

    env = {
        k: v
        for k, v in os.environ.items()
        if k != "WORKSPACE_MCP_GMAIL_PERMANENT_DELETE"
    }
    code = (
        "import gmail.gmail_admin_tools as a, auth.scopes as s;"
        "print(hasattr(a, 'delete_gmail_permanently'), s.GMAIL_FULL_SCOPE in s.GMAIL_SCOPES)"
    )
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True
    )
    assert out.stdout.strip() == "False False", out.stderr


@pytest.mark.asyncio
async def test_permanent_delete_requires_confirmation(monkeypatch):
    monkeypatch.setenv("WORKSPACE_MCP_GMAIL_PERMANENT_DELETE", "1")
    mod = importlib.reload(admin)
    try:
        fn = _unwrap(mod.delete_gmail_permanently)
        svc = MagicMock()
        with pytest.raises(UserInputError):
            await fn(service=svc, user_google_email=U, message_ids=["m1"])
        svc.users().messages().batchDelete.assert_not_called()

        ids = [f"m{i}" for i in range(1500)]
        out = await fn(
            service=svc, user_google_email=U, confirm_permanent=True, message_ids=ids
        )
        sizes = [
            len(c.kwargs["body"]["ids"])
            for c in svc.users().messages().batchDelete.call_args_list
            if "body" in c.kwargs
        ]
        assert sizes == [1000, 500]
        assert "Permanently deleted 1500 message(s)" in out
    finally:
        monkeypatch.delenv("WORKSPACE_MCP_GMAIL_PERMANENT_DELETE")
        importlib.reload(admin)


# --- vacation ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vacation_enable_builds_body():
    svc = MagicMock()
    svc.users().settings().getVacation().execute.return_value = {
        "enableAutoReply": True,
        "responseSubject": "Ferie",
    }
    out = await _unwrap(admin.manage_gmail_vacation)(
        service=svc,
        user_google_email=U,
        action="enable",
        subject="Ferie",
        body="Torno il 10",
        start_date="2026-10-01",
        end_date="2026-10-09",
    )
    body = svc.users().settings().updateVacation.call_args.kwargs["body"]
    assert body["enableAutoReply"] is True
    assert body["responseBodyPlainText"] == "Torno il 10"
    assert int(body["endTime"]) > int(body["startTime"])
    # end date is inclusive: ends at 23:59:59 of that day
    assert int(body["endTime"]) - int(body["startTime"]) > 8 * 86400 * 1000
    assert "Auto-reply: ON" in out


@pytest.mark.asyncio
async def test_vacation_enable_requires_body():
    with pytest.raises(UserInputError):
        await _unwrap(admin.manage_gmail_vacation)(
            service=MagicMock(), user_google_email=U, action="enable"
        )


@pytest.mark.asyncio
async def test_vacation_disable_keeps_other_fields():
    svc = MagicMock()
    svc.users().settings().getVacation().execute.return_value = {
        "enableAutoReply": True,
        "responseSubject": "Ferie",
    }
    await _unwrap(admin.manage_gmail_vacation)(
        service=svc, user_google_email=U, action="disable"
    )
    body = svc.users().settings().updateVacation.call_args.kwargs["body"]
    assert body == {"enableAutoReply": False, "responseSubject": "Ferie"}


def test_bad_date_is_user_error():
    with pytest.raises(UserInputError):
        admin._to_epoch_ms("01/10/2026")


# --- send-as / forwarding ------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_as_update_patches_only_given_fields():
    svc = MagicMock()
    svc.users().settings().sendAs().patch().execute.return_value = {
        "sendAsEmail": U,
        "signature": "<b>Marco</b>",
    }
    out = await _unwrap(admin.manage_gmail_send_as)(
        service=svc,
        user_google_email=U,
        action="update",
        send_as_email=U,
        signature="<b>Marco</b>",
    )
    kw = svc.users().settings().sendAs().patch.call_args.kwargs
    assert kw["body"] == {"signature": "<b>Marco</b>"}
    assert "Updated." in out


@pytest.mark.asyncio
async def test_send_as_requires_address():
    with pytest.raises(UserInputError):
        await _unwrap(admin.manage_gmail_send_as)(
            service=MagicMock(), user_google_email=U, action="get"
        )


@pytest.mark.asyncio
async def test_forwarding_enable_requires_address():
    with pytest.raises(UserInputError):
        await _unwrap(admin.manage_gmail_forwarding)(
            service=MagicMock(), user_google_email=U, action="enable_auto"
        )


@pytest.mark.asyncio
async def test_forwarding_enable_sets_disposition():
    svc = MagicMock()
    svc.users().settings().getAutoForwarding().execute.return_value = {
        "enabled": True,
        "emailAddress": "z@z.it",
        "disposition": "archive",
    }
    out = await _unwrap(admin.manage_gmail_forwarding)(
        service=svc,
        user_google_email=U,
        action="enable_auto",
        email_address="z@z.it",
        disposition="archive",
    )
    body = svc.users().settings().updateAutoForwarding.call_args.kwargs["body"]
    assert body == {"enabled": True, "emailAddress": "z@z.it", "disposition": "archive"}
    assert "ON → z@z.it" in out


# --- threads search -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_threads_formats_headers():
    svc = MagicMock()
    svc.users().threads().list().execute.return_value = {
        "threads": [{"id": "t1", "snippet": "ciao"}],
        "nextPageToken": "n2",
    }
    svc.users().threads().get().execute.return_value = {
        "messages": [
            {
                "labelIds": ["INBOX"],
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Oggetto"},
                        {"name": "From", "value": "a@b.it"},
                        {"name": "Date", "value": "D1"},
                    ]
                },
            },
            {
                "labelIds": ["SENT"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "me@x.it"},
                        {"name": "Date", "value": "D2"},
                    ]
                },
            },
        ]
    }
    out = await _unwrap(admin.search_gmail_threads)(
        service=svc, user_google_email=U, query="label:x"
    )
    assert "Thread ID: t1" in out
    assert "Subject: Oggetto" in out
    assert "Messages: 2 | Last: D2" in out
    assert "a@b.it; me@x.it" in out
    assert "INBOX, SENT" in out
    assert "next_page_token: n2" in out
