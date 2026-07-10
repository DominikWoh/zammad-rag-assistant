# 🔒 Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.0.x   | ✅ Active          |
| 1.x     | ❌ Legacy (`zammad-rag-assistant`) |

## Reporting a Vulnerability

**Please do not open public issues for security vulnerabilities.**

Instead, email: **security@zammad-lightrag.local** (or use GitHub's [private vulnerability reporting](https://github.com/duftertyp/zammad-lightrag/security/advisories/new)).

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You'll get a response within 72 hours. After confirmation, we aim to release a patch within 7 days.

## Security Best Practices for Users

- 🔐 **Never expose port 9621 to the public internet** without authentication
- 🔐 **Keep `.env` private** — it contains your Zammad API token
- 🔐 **Use a reverse proxy** (Traefik, Caddy, nginx) with TLS + auth if remote access is needed
- 🔐 **Rotate Zammad tokens** periodically
- 🔐 **Run containers as non-root** in production (work in progress)
- 🔐 **Pull images regularly** to get security updates
