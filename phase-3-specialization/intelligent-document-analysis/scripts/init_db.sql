-- Database initialization script for Intelligent Document Analysis System

-- Create the main database (if not exists)
-- Note: The database is already created by the POSTGRES_DB environment variable

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance
-- These will be created when the tables are created by SQLAlchemy

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE document_analysis TO postgres;

-- Create a simple test table to verify the database is working
CREATE TABLE IF NOT EXISTS health_check (
    id SERIAL PRIMARY KEY,
    check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'healthy'
);

-- Insert a test record
INSERT INTO health_check (status) VALUES ('database_initialized') ON CONFLICT DO NOTHING;

-- Display success message
SELECT 'Database initialized successfully' as message;
