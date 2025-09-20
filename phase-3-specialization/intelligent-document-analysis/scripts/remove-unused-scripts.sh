#!/bin/bash

# Remove Unused Scripts
# This script removes scripts that are no longer needed in production

set -e

echo "🗑️ Removing unused scripts..."

# Change to project directory
cd "$(dirname "$0")/.."

# Remove scripts that are no longer needed
echo "Removing comprehensive-fix-deploy.sh (one-time fix script)..."
rm -f scripts/comprehensive-fix-deploy.sh

echo "Removing optimize-for-azure.sh (optimization already applied)..."
rm -f scripts/optimize-for-azure.sh

echo "Removing start-azure.sh (startup handled by Dockerfile)..."
rm -f scripts/start-azure.sh

echo "Removing start.sh (local development script, not needed for Azure)..."
rm -f scripts/start.sh

echo "Removing init_db.sql (database initialization handled by code)..."
rm -f scripts/init_db.sql

echo "✅ Unused scripts removed successfully!"
echo ""
echo "📊 Remaining essential scripts:"
echo "  - deploy-azure.sh (main deployment script)"
echo "  - verify-deployment.sh (deployment verification)"
echo "  - health_check.py (health check utility)"
echo "  - housekeeping.sh (cleanup utility)"
echo ""
echo "🎯 Project is now optimized with only essential scripts!"
