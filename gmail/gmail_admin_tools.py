"""
Gmail mailbox administration tools.

Covers the parts of the Gmail API that gmail_tools.py does not expose:
thread search and thread-level label changes, trash / untrash and (opt-in)
permanent deletion, filter update and "apply filter to existing mail",
vacation responder, Send-As aliases and signatures, forwarding.

Kept in a separate module so the fork diff against upstream gmail_tools.py
stays small.
"""

import asyncio
import logging
from datetime import date, datetime, time as dt_time
from typing import Any, Dict, List, Literal, Optional

from mcp.types import ToolAnnotations

from auth.scopes import GMAIL_MODIFY_SCOPE, is_gmail_permanent_delete_enabled
from auth.service_decorator import require_google_service
from core.server import server
from core.utils import JsonDict, StringList, UserInputError, handle_http_errors

logger = logging.getLogger(__name__)

# users.messages.batchModify / batchDelete accept at most 1000 IDs per call.
_BATCH_LIMIT = 1000


def _chunks(items: List[str], size: int = _BATCH_LIMIT):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _header(headers: List[Dict[str, str]], name: str) -> str:
    for h in headers or []:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


@server.tool(
    title="Search Gmail Threads",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@handle_http_errors("search_gmail_threads", is_read_only=True, service_type="gmail")
@require_google_service("gmail", "gmail_read")
async def search_gmail_threads(
    service,
    user_google_email: str,
    query: str,
    page_size: int = 10,
    page_token: Optional[str] = None,
    include_headers: bool = True,
) -> str:
    """
    Searches Gmail conversations (threads) instead of single messages.
    Supports standard Gmail search operators (from:, to:, subject:, label:,
    is:unread, in:drafts, older_than:, has:attachment, ...).

    Args:
        user_google_email (str): The user's Google email address. Required.
        query (str): Gmail search query.
        page_size (int): Maximum threads to return (default 10, max 100).
        page_token (Optional[str]): Token from a previous call for the next page.
        include_headers (bool): Also fetch subject, participants, message count
            and last date for each thread (one metadata call per thread).

    Returns:
        str: Thread IDs with snippet (and headers), plus next_page_token.
    """
    params: Dict[str, Any] = {
        "userId": "me",
        "q": query,
        "maxResults": max(1, min(int(page_size), 100)),
    }
    if page_token:
        params["pageToken"] = page_token
    response = await asyncio.to_thread(service.users().threads().list(**params).execute)
    threads = response.get("threads") or []
    if not threads:
        return f"No threads found for query: '{query}'"

    details: Dict[str, Dict[str, Any]] = {}
    if include_headers:

        async def _meta(tid: str):
            return tid, await asyncio.to_thread(
                service.users()
                .threads()
                .get(
                    userId="me",
                    id=tid,
                    format="metadata",
                    metadataHeaders=["Subject", "From", "To", "Date"],
                )
                .execute
            )

        results = await asyncio.gather(
            *[_meta(t["id"]) for t in threads], return_exceptions=True
        )
        for res in results:
            if isinstance(res, Exception):
                continue
            tid, data = res
            details[tid] = data

    lines = [f"Found {len(threads)} threads matching '{query}':", ""]
    for i, t in enumerate(threads, 1):
        tid = t["id"]
        lines.append(f"  {i}. Thread ID: {tid}")
        data = details.get(tid)
        if data:
            msgs = data.get("messages") or []
            first = (msgs[0].get("payload") or {}).get("headers") if msgs else []
            last = (msgs[-1].get("payload") or {}).get("headers") if msgs else []
            senders = []
            for m in msgs:
                frm = _header((m.get("payload") or {}).get("headers"), "From")
                if frm and frm not in senders:
                    senders.append(frm)
            labels = sorted({lbl for m in msgs for lbl in (m.get("labelIds") or [])})
            lines.append(f"     Subject: {_header(first, 'Subject')}")
            lines.append(f"     Messages: {len(msgs)} | Last: {_header(last, 'Date')}")
            lines.append(f"     From: {'; '.join(senders[:5])}")
            lines.append(f"     Labels: {', '.join(labels)}")
        if t.get("snippet"):
            lines.append(f"     Snippet: {t['snippet']}")
        lines.append(f"     Link: https://mail.google.com/mail/u/0/#all/{tid}")
    if response.get("nextPageToken"):
        lines += ["", f"next_page_token: {response['nextPageToken']}"]
    return "\n".join(lines)


@server.tool(
    title="Modify Gmail Thread Labels",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@handle_http_errors("modify_gmail_thread_labels", service_type="gmail")
@require_google_service("gmail", GMAIL_MODIFY_SCOPE)
async def modify_gmail_thread_labels(
    service,
    user_google_email: str,
    thread_ids: StringList,
    add_label_ids: Optional[StringList] = None,
    remove_label_ids: Optional[StringList] = None,
) -> str:
    """
    Adds or removes labels on WHOLE conversations (every message in each thread).
    Archive a conversation: remove_label_ids=["INBOX"]. Mark read: remove "UNREAD".
    To trash conversations use manage_gmail_trash instead of the TRASH label.

    Args:
        user_google_email (str): The user's Google email address. Required.
        thread_ids (List[str]): Thread IDs to modify.
        add_label_ids (Optional[List[str]]): Label IDs to add.
        remove_label_ids (Optional[List[str]]): Label IDs to remove.

    Returns:
        str: Per-thread result.
    """
    if not thread_ids:
        raise UserInputError("thread_ids must not be empty.")
    if not add_label_ids and not remove_label_ids:
        raise UserInputError(
            "At least one of add_label_ids or remove_label_ids must be provided."
        )
    body: Dict[str, Any] = {}
    if add_label_ids:
        body["addLabelIds"] = list(add_label_ids)
    if remove_label_ids:
        body["removeLabelIds"] = list(remove_label_ids)

    ok, failed = [], []
    for tid in thread_ids:
        try:
            await asyncio.to_thread(
                service.users().threads().modify(userId="me", id=tid, body=body).execute
            )
            ok.append(tid)
        except Exception as e:  # keep going, report per thread
            failed.append(f"{tid}: {e}")
    lines = [f"Modified {len(ok)}/{len(thread_ids)} threads."]
    if add_label_ids:
        lines.append(f"Added: {', '.join(add_label_ids)}")
    if remove_label_ids:
        lines.append(f"Removed: {', '.join(remove_label_ids)}")
    if failed:
        lines.append("Failed:")
        lines += [f"  - {f}" for f in failed]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trash / untrash / permanent delete
# ---------------------------------------------------------------------------


@server.tool(
    title="Manage Gmail Trash",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@handle_http_errors("manage_gmail_trash", service_type="gmail")
@require_google_service("gmail", GMAIL_MODIFY_SCOPE)
async def manage_gmail_trash(
    service,
    user_google_email: str,
    action: Literal["trash", "untrash"],
    message_ids: Optional[StringList] = None,
    thread_ids: Optional[StringList] = None,
) -> str:
    """
    Moves messages or whole conversations to the Trash, or restores them.
    Trashed items are deleted by Gmail automatically after 30 days.
    Do NOT use this for drafts: use draft_gmail_message(action='delete').

    Args:
        user_google_email (str): The user's Google email address. Required.
        action (str): "trash" or "untrash".
        message_ids (Optional[List[str]]): Message IDs.
        thread_ids (Optional[List[str]]): Thread IDs (acts on every message).

    Returns:
        str: Per-item result.
    """
    if not message_ids and not thread_ids:
        raise UserInputError("Provide message_ids and/or thread_ids.")
    if action not in ("trash", "untrash"):
        raise UserInputError("action must be 'trash' or 'untrash'.")

    users = service.users()
    ok, failed = [], []
    for kind, ids, res in (
        ("message", message_ids or [], users.messages()),
        ("thread", thread_ids or [], users.threads()),
    ):
        for item_id in ids:
            try:
                call = getattr(res, action)(userId="me", id=item_id)
                await asyncio.to_thread(call.execute)
                ok.append(f"{kind} {item_id}")
            except Exception as e:
                failed.append(f"{kind} {item_id}: {e}")

    verb = "Trashed" if action == "trash" else "Restored"
    lines = [f"{verb} {len(ok)} item(s)."]
    lines += [f"  ✓ {x}" for x in ok[:50]]
    if len(ok) > 50:
        lines.append(f"  … (+{len(ok) - 50} more)")
    if failed:
        lines.append("Failed:")
        lines += [f"  ✗ {f}" for f in failed]
    return "\n".join(lines)


if is_gmail_permanent_delete_enabled():

    @server.tool(
        title="Delete Gmail Permanently",
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    @handle_http_errors("delete_gmail_permanently", service_type="gmail")
    @require_google_service("gmail", "gmail_full")
    async def delete_gmail_permanently(
        service,
        user_google_email: str,
        confirm_permanent: bool = False,
        message_ids: Optional[StringList] = None,
        thread_ids: Optional[StringList] = None,
    ) -> str:
        """
        PERMANENTLY deletes messages or whole conversations. This skips the Trash
        and CANNOT be undone. Prefer manage_gmail_trash(action='trash') unless the
        user explicitly asked for permanent deletion. Requires confirm_permanent=true.

        Args:
            user_google_email (str): The user's Google email address. Required.
            confirm_permanent (bool): Must be true, after the user confirmed.
            message_ids (Optional[List[str]]): Message IDs to delete forever.
            thread_ids (Optional[List[str]]): Thread IDs to delete forever.

        Returns:
            str: What was deleted.
        """
        if not confirm_permanent:
            raise UserInputError(
                "Refusing permanent deletion: set confirm_permanent=true only after "
                "the user explicitly confirmed. To make it recoverable use "
                "manage_gmail_trash(action='trash')."
            )
        if not message_ids and not thread_ids:
            raise UserInputError("Provide message_ids and/or thread_ids.")

        users = service.users()
        deleted_msgs = 0
        for chunk in _chunks(list(message_ids or [])):
            await asyncio.to_thread(
                users.messages().batchDelete(userId="me", body={"ids": chunk}).execute
            )
            deleted_msgs += len(chunk)
        ok_threads, failed = [], []
        for tid in thread_ids or []:
            try:
                await asyncio.to_thread(
                    users.threads().delete(userId="me", id=tid).execute
                )
                ok_threads.append(tid)
            except Exception as e:
                failed.append(f"thread {tid}: {e}")
        lines = [
            f"Permanently deleted {deleted_msgs} message(s) and "
            f"{len(ok_threads)} thread(s)."
        ]
        if failed:
            lines.append("Failed:")
            lines += [f"  ✗ {f}" for f in failed]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Filters: update and apply to existing mail
# ---------------------------------------------------------------------------


def _quote_term(value: str) -> str:
    value = str(value).strip()
    return f"({value})" if " " in value and not value.startswith("(") else value


def filter_criteria_to_query(criteria: Dict[str, Any]) -> str:
    """Translate Gmail filter criteria into an equivalent search query."""
    parts: List[str] = []
    if criteria.get("from"):
        parts.append(f"from:{_quote_term(criteria['from'])}")
    if criteria.get("to"):
        parts.append(f"to:{_quote_term(criteria['to'])}")
    if criteria.get("subject"):
        parts.append(f"subject:{_quote_term(criteria['subject'])}")
    if criteria.get("query"):
        parts.append(f"({criteria['query']})")
    if criteria.get("negatedQuery"):
        parts.append(f"-({criteria['negatedQuery']})")
    if criteria.get("hasAttachment"):
        parts.append("has:attachment")
    if criteria.get("size"):
        op = "larger" if criteria.get("sizeComparison") == "larger" else "smaller"
        parts.append(f"{op}:{int(criteria['size'])}")
    return " ".join(parts)


@server.tool(
    title="Update Gmail Filter",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
@handle_http_errors("update_gmail_filter", service_type="gmail")
@require_google_service("gmail", "gmail_settings_basic")
async def update_gmail_filter(
    service,
    user_google_email: str,
    filter_id: str,
    criteria: Optional[JsonDict] = None,
    filter_action: Optional[JsonDict] = None,
) -> str:
    """
    Changes an existing filter (auto-label rule). Gmail has no filter update API,
    so this creates the new filter FIRST and then deletes the old one: the filter
    ID changes. Fields you omit are kept from the old filter; pass a whole
    criteria or filter_action object to replace that part.

    Args:
        user_google_email (str): The user's Google email address. Required.
        filter_id (str): ID of the filter to change.
        criteria (Optional[Dict]): New criteria (from, to, subject, query,
            negatedQuery, hasAttachment, excludeChats, size, sizeComparison).
        filter_action (Optional[Dict]): New action (addLabelIds, removeLabelIds,
            forward).

    Returns:
        str: Old and new filter.
    """
    if not criteria and not filter_action:
        raise UserInputError("Provide criteria and/or filter_action to change.")
    filters = service.users().settings().filters()
    old = await asyncio.to_thread(filters.get(userId="me", id=filter_id).execute)
    body = {
        "criteria": dict(criteria) if criteria else old.get("criteria", {}),
        "action": dict(filter_action) if filter_action else old.get("action", {}),
    }
    created = await asyncio.to_thread(filters.create(userId="me", body=body).execute)
    await asyncio.to_thread(filters.delete(userId="me", id=filter_id).execute)
    return (
        "Filter updated (recreated).\n"
        f"Old ID: {filter_id} (deleted)\n"
        f"New ID: {created.get('id')}\n"
        f"Criteria: {created.get('criteria', body['criteria'])}\n"
        f"Action: {created.get('action', body['action'])}"
    )


@server.tool(
    title="Apply Gmail Filter To Existing Mail",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@handle_http_errors("apply_gmail_filter", service_type="gmail")
@require_google_service("gmail", ["gmail_settings_basic", GMAIL_MODIFY_SCOPE])
async def apply_gmail_filter(
    service,
    user_google_email: str,
    filter_id: Optional[str] = None,
    criteria: Optional[JsonDict] = None,
    filter_action: Optional[JsonDict] = None,
    dry_run: bool = False,
    max_messages: int = 5000,
) -> str:
    """
    Applies a filter's label actions to mail ALREADY in the mailbox (Gmail only
    applies filters to new mail; this is the web UI's "also apply to matching
    conversations"). Uses an existing filter_id, or ad-hoc criteria +
    filter_action. Forwarding is never applied retroactively.
    Use dry_run=true first to see the query and how many messages match.

    Args:
        user_google_email (str): The user's Google email address. Required.
        filter_id (Optional[str]): Existing filter to apply.
        criteria (Optional[Dict]): Criteria, if no filter_id.
        filter_action (Optional[Dict]): Action (addLabelIds/removeLabelIds), if no filter_id.
        dry_run (bool): Only count matches, change nothing.
        max_messages (int): Safety cap on messages changed (default 5000).

    Returns:
        str: Query used, matches, labels applied.
    """
    if filter_id:
        f = await asyncio.to_thread(
            service.users().settings().filters().get(userId="me", id=filter_id).execute
        )
        criteria, filter_action = f.get("criteria", {}), f.get("action", {})
    if not criteria or not filter_action:
        raise UserInputError("Provide filter_id, or both criteria and filter_action.")

    query = filter_criteria_to_query(criteria)
    if not query:
        raise UserInputError("Filter criteria produce an empty query; refusing.")
    add = list(filter_action.get("addLabelIds") or [])
    remove = list(filter_action.get("removeLabelIds") or [])
    notes = []
    if filter_action.get("forward"):
        notes.append("Forwarding is not applied to existing mail.")
    if criteria.get("excludeChats"):
        notes.append("excludeChats has no search equivalent and was ignored.")

    ids: List[str] = []
    page_token = None
    while len(ids) < max_messages:
        params = {"userId": "me", "q": query, "maxResults": 500}
        if page_token:
            params["pageToken"] = page_token
        resp = await asyncio.to_thread(
            service.users().messages().list(**params).execute
        )
        ids += [m["id"] for m in resp.get("messages") or []]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    truncated = len(ids) > max_messages or bool(page_token)
    ids = ids[:max_messages]

    lines = [
        f"Query: {query}",
        f"Matching messages: {len(ids)}{'+' if truncated else ''}",
    ]
    if add:
        lines.append(f"Add labels: {', '.join(add)}")
    if remove:
        lines.append(f"Remove labels: {', '.join(remove)}")
    lines += notes
    if dry_run:
        lines.insert(0, "DRY RUN — nothing changed.")
        return "\n".join(lines)
    if not add and not remove:
        lines.insert(0, "Nothing to apply: the filter has no label actions.")
        return "\n".join(lines)

    body: Dict[str, Any] = {}
    if add:
        body["addLabelIds"] = add
    if remove:
        body["removeLabelIds"] = remove
    for chunk in _chunks(ids):
        await asyncio.to_thread(
            service.users()
            .messages()
            .batchModify(userId="me", body={"ids": chunk, **body})
            .execute
        )
    lines.insert(0, f"Applied to {len(ids)} messages.")
    if truncated:
        lines.append(f"Stopped at max_messages={max_messages}; run again for the rest.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vacation responder
# ---------------------------------------------------------------------------


def _to_epoch_ms(value: str, end_of_day: bool = False) -> int:
    """Parse 'YYYY-MM-DD' or ISO datetime (local time if naive) to epoch ms."""
    value = value.strip()
    try:
        if len(value) == 10:
            d = date.fromisoformat(value)
            dt = datetime.combine(d, dt_time.max if end_of_day else dt_time.min)
        else:
            dt = datetime.fromisoformat(value)
    except ValueError as e:
        raise UserInputError(
            f"Invalid date '{value}': use YYYY-MM-DD or ISO 8601."
        ) from e
    if dt.tzinfo is None:
        dt = dt.astimezone()  # system local timezone
    return int(dt.timestamp() * 1000)


def _fmt_ms(ms: Optional[str]) -> str:
    if not ms:
        return "(none)"
    return (
        datetime.fromtimestamp(int(ms) / 1000)
        .astimezone()
        .isoformat(timespec="minutes")
    )


@server.tool(
    title="Manage Gmail Vacation Responder",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@handle_http_errors("manage_gmail_vacation", service_type="gmail")
@require_google_service("gmail", "gmail_settings_basic")
async def manage_gmail_vacation(
    service,
    user_google_email: str,
    action: Literal["get", "enable", "disable"],
    subject: Optional[str] = None,
    body: Optional[str] = None,
    body_format: Literal["plain", "html"] = "plain",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    restrict_to_contacts: bool = False,
    restrict_to_domain: bool = False,
) -> str:
    """
    Reads, enables or disables the vacation auto-reply (out of office).

    Args:
        user_google_email (str): The user's Google email address. Required.
        action (str): "get", "enable" or "disable".
        subject (Optional[str]): Auto-reply subject (enable).
        body (Optional[str]): Auto-reply text (enable; required).
        body_format (str): "plain" or "html".
        start_date (Optional[str]): YYYY-MM-DD or ISO datetime; from 00:00 local.
        end_date (Optional[str]): YYYY-MM-DD or ISO datetime; until 23:59 local.
        restrict_to_contacts (bool): Reply only to people in Contacts.
        restrict_to_domain (bool): Reply only within the domain (Workspace).

    Returns:
        str: Current vacation settings.
    """
    settings = service.users().settings()
    if action == "enable":
        if not body:
            raise UserInputError("body is required to enable the auto-reply.")
        cfg: Dict[str, Any] = {
            "enableAutoReply": True,
            "responseSubject": subject or "",
            "restrictToContacts": restrict_to_contacts,
            "restrictToDomain": restrict_to_domain,
        }
        if body_format == "html":
            cfg["responseBodyHtml"] = body
        else:
            cfg["responseBodyPlainText"] = body
        if start_date:
            cfg["startTime"] = str(_to_epoch_ms(start_date))
        if end_date:
            cfg["endTime"] = str(_to_epoch_ms(end_date, end_of_day=True))
        await asyncio.to_thread(settings.updateVacation(userId="me", body=cfg).execute)
    elif action == "disable":
        current = await asyncio.to_thread(settings.getVacation(userId="me").execute)
        current["enableAutoReply"] = False
        await asyncio.to_thread(
            settings.updateVacation(userId="me", body=current).execute
        )
    elif action != "get":
        raise UserInputError("action must be 'get', 'enable' or 'disable'.")

    v = await asyncio.to_thread(settings.getVacation(userId="me").execute)
    return "\n".join(
        [
            f"Auto-reply: {'ON' if v.get('enableAutoReply') else 'OFF'}",
            f"Subject: {v.get('responseSubject', '')}",
            f"Body: {v.get('responseBodyPlainText') or v.get('responseBodyHtml') or ''}",
            f"Start: {_fmt_ms(v.get('startTime'))}",
            f"End: {_fmt_ms(v.get('endTime'))}",
            f"Only contacts: {v.get('restrictToContacts', False)}",
            f"Only domain: {v.get('restrictToDomain', False)}",
        ]
    )


# ---------------------------------------------------------------------------
# Send-As aliases and signatures
# ---------------------------------------------------------------------------


def _fmt_send_as(s: Dict[str, Any]) -> str:
    flags = []
    if s.get("isPrimary"):
        flags.append("primary")
    if s.get("isDefault"):
        flags.append("default")
    if s.get("verificationStatus"):
        flags.append(s["verificationStatus"])
    sig = s.get("signature") or ""
    return "\n".join(
        [
            f"🔹 {s.get('sendAsEmail')} ({', '.join(flags) or '-'})",
            f"   Display name: {s.get('displayName', '')}",
            f"   Reply-To: {s.get('replyToAddress', '') or '(none)'}",
            f"   Signature (HTML): {sig or '(none)'}",
        ]
    )


@server.tool(
    title="Manage Gmail Send-As and Signatures",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
@handle_http_errors("manage_gmail_send_as", service_type="gmail")
@require_google_service("gmail", ["gmail_settings_basic", "gmail_settings_sharing"])
async def manage_gmail_send_as(
    service,
    user_google_email: str,
    action: Literal["list", "get", "update", "create", "delete", "verify"],
    send_as_email: Optional[str] = None,
    display_name: Optional[str] = None,
    reply_to_address: Optional[str] = None,
    signature: Optional[str] = None,
    is_default: Optional[bool] = None,
    treat_as_alias: Optional[bool] = None,
) -> str:
    """
    Manages "Send mail as" addresses and their signatures.
    Change the signature: action="update", send_as_email=<address>, signature=<HTML>.
    create adds an alias (Gmail emails a verification link to it); verify re-sends it.

    Args:
        user_google_email (str): The user's Google email address. Required.
        action (str): list, get, update, create, delete, verify.
        send_as_email (Optional[str]): The alias address (all but list).
        display_name (Optional[str]): Name shown in From.
        reply_to_address (Optional[str]): Reply-To address ("" to clear).
        signature (Optional[str]): Signature HTML ("" to clear).
        is_default (Optional[bool]): Make this the default From address.
        treat_as_alias (Optional[bool]): Gmail "treat as alias" option.

    Returns:
        str: The resulting Send-As settings.
    """
    send_as = service.users().settings().sendAs()
    if action == "list":
        resp = await asyncio.to_thread(send_as.list(userId="me").execute)
        items = resp.get("sendAs") or []
        return "\n\n".join(_fmt_send_as(s) for s in items) or "No Send-As addresses."
    if not send_as_email:
        raise UserInputError("send_as_email is required for this action.")

    fields: Dict[str, Any] = {}
    if display_name is not None:
        fields["displayName"] = display_name
    if reply_to_address is not None:
        fields["replyToAddress"] = reply_to_address
    if signature is not None:
        fields["signature"] = signature
    if is_default is not None:
        fields["isDefault"] = is_default
    if treat_as_alias is not None:
        fields["treatAsAlias"] = treat_as_alias

    if action == "get":
        s = await asyncio.to_thread(
            send_as.get(userId="me", sendAsEmail=send_as_email).execute
        )
        return _fmt_send_as(s)
    if action == "update":
        if not fields:
            raise UserInputError("Nothing to update.")
        s = await asyncio.to_thread(
            send_as.patch(userId="me", sendAsEmail=send_as_email, body=fields).execute
        )
        return "Updated.\n" + _fmt_send_as(s)
    if action == "create":
        s = await asyncio.to_thread(
            send_as.create(
                userId="me", body={"sendAsEmail": send_as_email, **fields}
            ).execute
        )
        return (
            "Created (check the alias inbox for Gmail's verification email).\n"
            + _fmt_send_as(s)
        )
    if action == "delete":
        await asyncio.to_thread(
            send_as.delete(userId="me", sendAsEmail=send_as_email).execute
        )
        return f"Deleted Send-As address {send_as_email}."
    if action == "verify":
        await asyncio.to_thread(
            send_as.verify(userId="me", sendAsEmail=send_as_email).execute
        )
        return f"Verification email re-sent to {send_as_email}."
    raise UserInputError(f"Unknown action '{action}'.")


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------


@server.tool(
    title="Manage Gmail Forwarding",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
@handle_http_errors("manage_gmail_forwarding", service_type="gmail")
@require_google_service("gmail", ["gmail_settings_basic", "gmail_settings_sharing"])
async def manage_gmail_forwarding(
    service,
    user_google_email: str,
    action: Literal[
        "list_addresses",
        "add_address",
        "remove_address",
        "get_auto",
        "enable_auto",
        "disable_auto",
    ],
    email_address: Optional[str] = None,
    disposition: Literal[
        "leaveInInbox", "archive", "trash", "markRead"
    ] = "leaveInInbox",
) -> str:
    """
    Manages forwarding addresses and automatic forwarding of all incoming mail.
    add_address makes Gmail send a confirmation link to that address; it must be
    confirmed before enable_auto can use it. (Per-filter forwarding goes in the
    filter's action.forward.)

    Args:
        user_google_email (str): The user's Google email address. Required.
        action (str): list_addresses, add_address, remove_address, get_auto,
            enable_auto, disable_auto.
        email_address (Optional[str]): Forwarding address (add/remove/enable_auto).
        disposition (str): What to do with the Gmail copy when auto-forwarding:
            leaveInInbox, archive, trash, markRead.

    Returns:
        str: Result and current state.
    """
    settings = service.users().settings()
    fwd = settings.forwardingAddresses()

    if action == "list_addresses":
        resp = await asyncio.to_thread(fwd.list(userId="me").execute)
        items = resp.get("forwardingAddresses") or []
        if not items:
            return "No forwarding addresses."
        return "\n".join(
            f"- {a.get('forwardingEmail')} ({a.get('verificationStatus')})"
            for a in items
        )
    if action in ("add_address", "remove_address", "enable_auto") and not email_address:
        raise UserInputError("email_address is required for this action.")
    if action == "add_address":
        a = await asyncio.to_thread(
            fwd.create(userId="me", body={"forwardingEmail": email_address}).execute
        )
        return (
            f"Added {a.get('forwardingEmail')} ({a.get('verificationStatus')}). "
            "Gmail sent a confirmation link to that address."
        )
    if action == "remove_address":
        await asyncio.to_thread(
            fwd.delete(userId="me", forwardingEmail=email_address).execute
        )
        return f"Removed forwarding address {email_address}."
    if action == "enable_auto":
        await asyncio.to_thread(
            settings.updateAutoForwarding(
                userId="me",
                body={
                    "enabled": True,
                    "emailAddress": email_address,
                    "disposition": disposition,
                },
            ).execute
        )
    elif action == "disable_auto":
        await asyncio.to_thread(
            settings.updateAutoForwarding(userId="me", body={"enabled": False}).execute
        )
    elif action != "get_auto":
        raise UserInputError(f"Unknown action '{action}'.")

    a = await asyncio.to_thread(settings.getAutoForwarding(userId="me").execute)
    if not a.get("enabled"):
        return "Auto-forwarding: OFF"
    return (
        f"Auto-forwarding: ON → {a.get('emailAddress')} "
        f"(Gmail copy: {a.get('disposition')})"
    )
