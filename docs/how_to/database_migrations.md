# How to manage database migrations

The persistence layer is database-agnostic — the backend is chosen purely by the SQLAlchemy async URL in `OMNIAI_DATABASE_URL`:

```bash
OMNIAI_DATABASE_URL=postgresql+asyncpg://omniai:pw@db:5432/omniai   # production
OMNIAI_DATABASE_URL=sqlite+aiosqlite:///interactions.db             # zero-config dev
# any other async dialect (e.g. mysql+asyncmy://) works unchanged
```

## Applying migrations

Alembic drives schema changes; `migrations/env.py` reads the same `OMNIAI_DATABASE_URL` as the app, using the async engine (asyncpg/aiosqlite — no sync driver needed).

```bash
alembic upgrade head        # apply pending migrations
alembic downgrade -1        # roll back one revision
alembic history             # list revisions
```

In the [Compose deployment](deploy_docker_compose.md) the container entrypoint runs `alembic upgrade head` automatically before starting uvicorn — deploys are always schema-current.

## Creating a new migration

After changing `omniai/memory/models.py`:

```bash
alembic revision --autogenerate -m "add my_column to interactions"
# review the generated file in migrations/versions/, then:
alembic upgrade head
```

Keep column types portable (plain `String`/`DateTime`, JSON as TEXT) — that's what keeps the layer database-agnostic.

## Dev vs production

`InteractionBuffer` also runs `create_all` on first use, so fresh dev databases work with no migration step; `create_all` never alters existing tables, so it coexists safely with Alembic. For a database that predates Alembic (created by `create_all` alone), baseline it once with `alembic stamp head`.
