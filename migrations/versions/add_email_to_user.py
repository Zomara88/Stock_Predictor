
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_email_to_user'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('user', sa.Column('email', sa.String(length=150), nullable=False, unique=True))

def downgrade():
    op.drop_column('user', 'email')