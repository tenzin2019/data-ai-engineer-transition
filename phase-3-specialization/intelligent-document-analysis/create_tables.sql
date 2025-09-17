-- Create tables for Intelligent Document Analysis System

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100),
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    is_superuser BOOLEAN DEFAULT FALSE NOT NULL,
    organization VARCHAR(100),
    role VARCHAR(50),
    last_login TIMESTAMP WITH TIME ZONE,
    email_verified BOOLEAN DEFAULT FALSE NOT NULL,
    email_verified_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for users
CREATE INDEX IF NOT EXISTS ix_users_id ON users (id);
CREATE INDEX IF NOT EXISTS ix_users_username ON users (username);
CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE
);

-- Create indexes for user_sessions
CREATE INDEX IF NOT EXISTS ix_user_sessions_id ON user_sessions (id);
CREATE INDEX IF NOT EXISTS ix_user_sessions_session_token ON user_sessions (session_token);
CREATE INDEX IF NOT EXISTS ix_user_sessions_refresh_token ON user_sessions (refresh_token);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    document_type VARCHAR(20) NOT NULL DEFAULT 'unknown',
    status VARCHAR(20) NOT NULL DEFAULT 'uploaded',
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    extracted_text TEXT,
    text_length INTEGER,
    page_count INTEGER,
    summary TEXT,
    key_phrases JSON,
    sentiment_score REAL,
    confidence_score REAL,
    document_metadata JSON,
    tags JSON,
    user_id INTEGER REFERENCES users(id)
);

-- Create indexes for documents
CREATE INDEX IF NOT EXISTS ix_documents_id ON documents (id);
CREATE INDEX IF NOT EXISTS ix_documents_filename ON documents (filename);

-- Document analyses table
CREATE TABLE IF NOT EXISTS document_analyses (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    analysis_type VARCHAR(100) NOT NULL,
    analysis_data JSON NOT NULL,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    processing_time REAL,
    token_count INTEGER,
    cost REAL,
    confidence_score REAL,
    quality_score REAL
);

-- Create indexes for document_analyses
CREATE INDEX IF NOT EXISTS ix_document_analyses_id ON document_analyses (id);

-- Document entities table
CREATE TABLE IF NOT EXISTS document_entities (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    entity_text VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_label VARCHAR(100),
    start_position INTEGER,
    end_position INTEGER,
    page_number INTEGER,
    confidence_score REAL,
    entity_metadata JSON
);

-- Create indexes for document_entities
CREATE INDEX IF NOT EXISTS ix_document_entities_id ON document_entities (id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON user_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_document_analyses_updated_at BEFORE UPDATE ON document_analyses FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_document_entities_updated_at BEFORE UPDATE ON document_entities FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert a test document
INSERT INTO documents (filename, original_filename, file_path, file_size, mime_type, document_type, status, extracted_text, text_length, summary) 
VALUES ('test_document.pdf', 'test_document.pdf', '/app/uploads/test_document.pdf', 1024, 'application/pdf', 'pdf', 'completed', 'This is a test document for the intelligent document analysis system.', 67, 'A test document for system verification.')
ON CONFLICT DO NOTHING;

-- Show created tables
SELECT 'Tables created successfully!' as message;
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
