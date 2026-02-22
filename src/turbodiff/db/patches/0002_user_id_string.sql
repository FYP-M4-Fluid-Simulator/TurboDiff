-- Patch 0002: Change user ID columns from UUID to TEXT to support Firebase UIDs

-- We must drop the foreign key constraints first
ALTER TABLE airfoils DROP CONSTRAINT airfoils_created_by_user_id_fkey;
ALTER TABLE sessions DROP CONSTRAINT sessions_user_id_fkey;

-- Change the columns to TEXT
ALTER TABLE users ALTER COLUMN id TYPE TEXT;
ALTER TABLE sessions ALTER COLUMN user_id TYPE TEXT;
ALTER TABLE airfoils ALTER COLUMN created_by_user_id TYPE TEXT;

-- Re-add the foreign key constraints
ALTER TABLE sessions ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE airfoils ADD CONSTRAINT airfoils_created_by_user_id_fkey FOREIGN KEY (created_by_user_id) REFERENCES users(id) ON DELETE SET NULL;
