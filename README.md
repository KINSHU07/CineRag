---

## Switching LLM

Just change one line in your `.env` file:
```env
ACTIVE_API=mistral    # or claude, openai, huggingface
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| Backend offline | Start uvicorn in a separate terminal |
| MongoDB auth failed | Encode @ as %40 in your password |
| 1024 vs 384 dimensions | Use thenlper/gte-large not all-MiniLM-L6-v2 |
| ModuleNotFoundError | pip install mistralai==1.2.5 inside your virtualenv |

---

## Author

Kinshu Keshri - [GitHub](https://github.com/yourusername)
