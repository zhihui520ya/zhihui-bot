"""
邮件工具 —— SMTP 发信 + IMAP 读信。
发信用 smtplib，读信用 imaplib（均为标准库）。
"""
import os
import re
import smtplib
import ssl
import imaplib
import email
from email.header import decode_header
from email.message import EmailMessage
from email.utils import formataddr, parsedate_to_datetime
from langchain.tools import tool

# SMTP/IMAP 配置（从 config.py 读取，通过 .env 设置）
from config import (
    SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
    SMTP_FROM_EMAIL, SMTP_FROM_NAME, SMTP_USE_TLS, SMTP_USE_STARTTLS,
    IMAP_HOST, IMAP_PORT,
)


def _smtp_send(
    to_addrs: list[str],
    msg: EmailMessage,
) -> tuple[int, str]:
    """底层 SMTP 发送，返回 (状态码, 状态消息)"""
    if SMTP_USE_TLS:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=15) as server:
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            return (250, "OK")
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            if SMTP_USE_STARTTLS:
                context = ssl.create_default_context()
                server.starttls(context=context)
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            return (250, "OK")


def _build_message(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    is_html: bool = False,
) -> EmailMessage:
    """构建邮件"""
    msg = EmailMessage()
    msg["From"] = formataddr((SMTP_FROM_NAME, SMTP_FROM_EMAIL))
    msg["To"] = to
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = cc

    if is_html:
        msg.set_content(body.replace("<br>", "\n").replace("<br/>", "\n")
                        .replace("</p>", "\n").replace("</div>", "\n")
                        .strip(), plain="")
        msg.add_alternative(body, subtype="html")
    else:
        msg.set_content(body)

    return msg


@tool
def send_email(
    to: str = "",
    subject: str = "",
    body: str = "",
    cc: str = "",
    is_html: bool = False,
) -> str:
    """
    发送电子邮件。当用户要求"发邮件"、"写邮件"、"发送到邮箱"、"给某人发邮件"时使用此工具。
    如果用户说"给我发邮件"或"发到我的邮箱"，你可以直接用他的 QQ 号 + @qq.com 作为收件人地址。

    Args:
        to: 收件人邮箱地址。如果传的是纯数字（QQ号），会自动补成 QQ号@qq.com。
            多个收件人用英文逗号分隔，如 "a@example.com,123456@qq.com"。
            可选，如果没提供则向用户询问。
        subject: 邮件主题
        body: 邮件正文内容
        cc: 抄送邮箱地址（可选），多个用英文逗号分隔
        is_html: 正文是否为 HTML 格式（可选，默认 False）。为 True 时 body 应包含 HTML 标签
    """
    if not SMTP_HOST or not SMTP_FROM_EMAIL:
        return "邮件服务未配置，请联系管理员设置 SMTP 配置。"

    # QQ 号自动补全邮箱
    if to and re.match(r'^\d+$', to.strip()):
        to = to.strip() + "@qq.com"

    if not to or not subject or not body:
        return "请提供完整的收件人地址、主题和正文。"

    print(f"[工具调用] 发送邮件: to={to}, subject={subject}")

    try:
        # 解析收件人
        to_addrs = []
        for addr in to.split(","):
            addr = addr.strip()
            if re.match(r'^\d+$', addr):
                addr += "@qq.com"
            to_addrs.append(addr)
        cc_addrs = [a.strip() for a in cc.split(",") if a.strip()] if cc else []
        all_recipients = to_addrs + cc_addrs

        msg = _build_message(to, subject, body, cc, is_html)
        _smtp_send(all_recipients, msg)

        result = f"邮件发送成功！收件人: {to}"
        if cc:
            result += f"，抄送: {cc}"
        result += f"，主题: {subject}"
        print(f"[工具结果] {result}")
        return result
    except smtplib.SMTPAuthenticationError:
        error_msg = "邮件发送失败：SMTP 认证失败，请检查用户名或授权码是否正确。"
        print(f"[工具错误] {error_msg}")
        return error_msg
    except smtplib.SMTPRecipientsRefused:
        error_msg = "邮件发送失败：收件人被服务器拒绝，请检查邮箱地址是否正确。"
        print(f"[工具错误] {error_msg}")
        return error_msg
    except smtplib.SMTPServerDisconnected:
        error_msg = "邮件发送失败：SMTP 服务器连接断开，请检查网络或服务器地址。"
        print(f"[工具错误] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"邮件发送失败：{str(e)}"
        print(f"[工具错误] {error_msg}")
        return error_msg


# ==================== IMAP 读信 ====================


def _decode_str(s: bytes | str | None, charset: str = "utf-8") -> str:
    """解码邮件头字段，处理各种编码"""
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode(charset, errors="replace")
    decoded_parts = []
    try:
        for part, enc in decode_header(s):
            if isinstance(part, bytes):
                decoded_parts.append(part.decode(enc or charset, errors="replace"))
            else:
                decoded_parts.append(part)
    except Exception:
        return s
    return " ".join(decoded_parts).strip()


def _decode_body(payload: bytes, charset: str = "utf-8") -> str:
    """解码邮件正文"""
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        # QQ 邮件常用 GBK/GB2312
        for cs in ("gbk", "gb2312", "utf-8"):
            try:
                return payload.decode(cs, errors="replace")
            except (LookupError, UnicodeDecodeError):
                continue
        return payload.decode("utf-8", errors="replace")


def _get_email_body(msg: email.message.Message) -> str:
    """递归提取邮件正文（优先纯文本）"""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload:
                    return _decode_body(payload, charset)
            elif content_type == "text/html":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload:
                    return _decode_body(payload, charset)
        return ""
    else:
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset() or "utf-8"
        if payload:
            return _decode_body(payload, charset)
        return ""


def _format_email_preview(i: int, msg: email.message.Message) -> str:
    """格式化邮件预览行"""
    subject = _decode_str(msg["Subject"])
    sender = _decode_str(msg.get("From", ""))
    date_str = msg.get("Date", "")
    # 尝试解析日期
    try:
        dt = parsedate_to_datetime(date_str)
        date_str = dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        date_str = date_str[:25]
    # 截断过长的主题和发件人
    if len(subject) > 40:
        subject = subject[:38] + "…"
    sender_clean = sender.split("<")[0].strip().strip('"\'') or sender
    if len(sender_clean) > 25:
        sender_clean = sender_clean[:23] + "…"
    return f"{i}. [{date_str}] {sender_clean}  —  {subject}"


def _imap_connect():
    """连接 IMAP 服务器并登录"""
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        raise ConnectionError("IMAP 未配置，请检查邮箱账号和授权码。")
    conn = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, timeout=15)
    conn.login(SMTP_USERNAME, SMTP_PASSWORD)
    return conn


@tool
def check_emails(limit: int = 5, folder: str = "INBOX") -> str:
    """
    查看收件箱最新邮件列表，返回每封邮件的序号、发件人、主题和收发时间。
    用户想"查邮件""看收件箱""有没有新邮件"时使用此工具。

    Args:
        limit: 要获取的邮件数量（默认5，最大20）
        folder: 邮箱文件夹（默认 INBOX）
    """
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        return "邮件服务未配置。"

    limit = min(max(limit, 1), 20)
    print(f"[工具调用] 查收邮件: folder={folder}, limit={limit}")

    try:
        conn = _imap_connect()
        try:
            conn.select(folder)
            _, data = conn.search(None, "ALL")
            if not data or not data[0]:
                return "收件箱为空。"

            msg_ids = data[0].split()
            # 取最新的 limit 封
            recent_ids = msg_ids[-limit:]
            results = []
            for mid in reversed(recent_ids):
                _, fetch_data = conn.fetch(mid, "(BODY.PEEK[HEADER])")
                if fetch_data and fetch_data[0]:
                    raw_header = fetch_data[0][1]
                    parsed = email.message_from_bytes(raw_header)
                    results.append(parsed)

            if not results:
                return "未能读取到邮件。"

            lines = [f"📬 {folder} 最新 {len(results)} 封邮件："]
            for i, parsed in enumerate(results, 1):
                lines.append(_format_email_preview(i, parsed))

            return "\n".join(lines)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            conn.logout()
    except imaplib.IMAP4.error as e:
        return f"IMAP 错误：{str(e)}"
    except ConnectionError as e:
        return str(e)
    except Exception as e:
        return f"读取邮件失败：{str(e)}"


@tool
def read_email(index: int = 1, folder: str = "INBOX") -> str:
    """
    读取收件箱中指定序号的邮件完整内容（发件人、时间、收件人、主题、正文）。
    index 对应 check_emails 返回的序号，1 为最新邮件。

    Args:
        index: 邮件序号（从1开始，1为最新）
        folder: 邮箱文件夹（默认 INBOX）
    """
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        return "邮件服务未配置。"

    index = max(index, 1)
    print(f"[工具调用] 阅读邮件: folder={folder}, index={index}")

    try:
        conn = _imap_connect()
        try:
            conn.select(folder)
            _, data = conn.search(None, "ALL")
            if not data or not data[0]:
                return "收件箱为空。"

            msg_ids = data[0].split()
            if index > len(msg_ids):
                return f"序号超出范围，收件箱只有 {len(msg_ids)} 封邮件。"

            # index=1 是最新，所以从末尾往前数
            target_id = msg_ids[-index]

            # 获取完整邮件（含正文）
            _, fetch_data = conn.fetch(target_id, "(BODY[])")
            if not fetch_data or not fetch_data[0]:
                return "无法读取该邮件。"

            raw_email = fetch_data[0][1]
            if isinstance(raw_email, bytes):
                parsed = email.message_from_bytes(raw_email)
            else:
                parsed = email.message_from_string(raw_email)

            subject = _decode_str(parsed["Subject"])
            sender = _decode_str(parsed.get("From", ""))
            recipient = _decode_str(parsed.get("To", ""))
            date_str = parsed.get("Date", "")
            try:
                dt = parsedate_to_datetime(date_str)
                date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                date_str = date_str[:30]
            body = _get_email_body(parsed)

            result = (
                f"📧 邮件详情\n"
                f"主题: {subject}\n"
                f"发件人: {sender}\n"
                f"收件人: {recipient}\n"
                f"时间: {date_str}\n"
                f"---正文---\n{body[:3000]}"
            )
            if len(body) > 3000:
                result += "\n…（正文过长已截断）"
            return result
        finally:
            try:
                conn.close()
            except Exception:
                pass
            conn.logout()
    except imaplib.IMAP4.error as e:
        return f"IMAP 错误：{str(e)}"
    except ConnectionError as e:
        return str(e)
    except Exception as e:
        return f"读取邮件失败：{str(e)}"
