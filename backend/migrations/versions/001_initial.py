"""Initial migration - create tables

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create analyses table
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column(
            "type",
            sa.Enum("image", "video", name="analysistype"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("pending", "processing", "completed", "failed", name="analysisstatus"),
            nullable=False,
        ),
        sa.Column(
            "verdict",
            sa.Enum("REAL", "FAKE", "UNCERTAIN", name="verdict"),
            nullable=True,
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("runtime_ms", sa.Integer(), nullable=True),
        sa.Column("device", sa.String(20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("upload_path", sa.String(500), nullable=True),
        sa.Column("total_frames", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_analyses_sha256"), "analyses", ["sha256"], unique=False)

    # Create frames table
    op.create_table(
        "frames",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.String(36), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("overlay_path", sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["analyses.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create assets table
    op.create_table(
        "assets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.String(36), nullable=False),
        sa.Column("kind", sa.String(50), nullable=False),
        sa.Column("path", sa.String(500), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["analyses.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("assets")
    op.drop_table("frames")
    op.drop_index(op.f("ix_analyses_sha256"), table_name="analyses")
    op.drop_table("analyses")
    
    # Drop enums
    op.execute("DROP TYPE IF EXISTS verdict")
    op.execute("DROP TYPE IF EXISTS analysisstatus")
    op.execute("DROP TYPE IF EXISTS analysistype")
