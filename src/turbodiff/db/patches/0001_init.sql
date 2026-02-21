-- Patch 0001: initial schema

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    email TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE cst (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    weights_upper JSONB NOT NULL,
    weights_lower JSONB NOT NULL,
    chord_length DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_type TEXT NOT NULL CHECK (session_type IN ('optimize', 'simulate')),
    parameters JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE airfoils (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cst_id UUID NOT NULL REFERENCES cst(id) ON DELETE RESTRICT,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    is_optimized BOOLEAN NOT NULL DEFAULT FALSE,
    cl DOUBLE PRECISION,
    cd DOUBLE PRECISION,
    lift DOUBLE PRECISION,
    drag DOUBLE PRECISION,
    angle_of_attack DOUBLE PRECISION,
    created_by_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX sessions_user_id_created_at_idx ON sessions (user_id, created_at DESC);
CREATE INDEX airfoils_user_id_created_at_idx ON airfoils (created_by_user_id, created_at DESC);
CREATE INDEX airfoils_session_id_idx ON airfoils (session_id);
