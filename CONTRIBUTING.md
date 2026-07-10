# 🤝 Contributing

Thanks for your interest in contributing to Zammad-LightRAG! 🎉

## How to Contribute

### 🐛 Reporting Bugs

Open an [Issue](https://github.com/duftertyp/zammad-lightrag/issues/new) with:

- **Clear title** describing the bug
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Environment**: OS, Docker version, LLM backend, relevant config
- **Logs**: `docker logs zammad-lightrag --tail 50` / `docker logs zammad-sync --tail 50`

### 💡 Suggesting Features

Open an [Issue](https://github.com/duftertyp/zammad-lightrag/issues/new) with the `enhancement` label. Describe:

- What problem does this solve?
- How should it work?
- Any alternatives you considered?

### 🔧 Submitting Code

1. **Fork** the repo
2. **Create a branch** from `master`:
   ```bash
   git checkout -b feature/awesome-thing
   ```
3. **Make your changes**
4. **Test locally**:
   ```bash
   docker compose build
   docker compose up -d
   # verify it works
   ```
5. **Commit** with a clear message:
   ```bash
   git commit -m "Add awesome-thing"
   ```
6. **Push** and open a Pull Request

## 📋 Style Guide

- **Python**: PEP 8, type hints where reasonable
- **Comments**: English in code, German/English in user-facing strings
- **Commits**: Imperative mood ("Add feature" not "Added feature")
- **No credentials**: Never commit `.env`, tokens, or private URLs

## 🧪 Testing

Before submitting a PR:

- [ ] `docker compose build` succeeds
- [ ] `docker compose up -d` starts both containers healthy
- [ ] Test sync: `docker exec zammad-sync python sync.py --once --limit=5`
- [ ] Test query: open `http://localhost:9621/webui`
- [ ] No secrets in your changes (run `git diff` to verify)

## 📜 License

By contributing, you agree that your contributions will be licensed under [MIT](LICENSE).
