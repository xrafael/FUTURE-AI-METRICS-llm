#!/bin/bash

# Script to create GitHub repository and push code
# Usage: ./create_github_repo.sh [GITHUB_TOKEN]

GITHUB_TOKEN=${1:-$GITHUB_TOKEN}
REPO_NAME="FUTURE-AI-METRICS-LLM"
USERNAME="xrafael"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GitHub token is required"
    echo "Usage: ./create_github_repo.sh YOUR_GITHUB_TOKEN"
    echo "Or set GITHUB_TOKEN environment variable"
    echo ""
    echo "To create a token:"
    echo "1. Go to https://github.com/settings/tokens"
    echo "2. Click 'Generate new token (classic)'"
    echo "3. Select 'repo' scope"
    echo "4. Copy the token and use it here"
    exit 1
fi

echo "Creating repository $REPO_NAME on GitHub..."

# Create the repository
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$REPO_NAME\",\"description\":\"LLM-based evaluation tools for assessing AI clinical universality metrics in medical imaging research papers\",\"private\":false}")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 201 ]; then
    echo "✓ Repository created successfully!"
    echo ""
    echo "Pushing code to GitHub..."
    git push -u origin main
    echo ""
    echo "✓ Done! Repository available at: https://github.com/$USERNAME/$REPO_NAME"
elif [ "$HTTP_CODE" -eq 422 ]; then
    echo "Repository may already exist. Attempting to push..."
    git push -u origin main
    if [ $? -eq 0 ]; then
        echo "✓ Code pushed successfully!"
        echo "Repository available at: https://github.com/$USERNAME/$REPO_NAME"
    else
        echo "Error: Failed to push. Please check your repository permissions."
    fi
else
    echo "Error creating repository:"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    exit 1
fi

