#!/bin/bash
# Simple API Testing Script
# Tests the Face Recognition Attendance System API endpoints

BASE_URL="http://localhost:3000"

echo "======================================================================"
echo "Face Recognition Attendance System - API Testing"
echo "======================================================================"
echo ""

# Check if server is running
echo "Testing server connection..."
if curl -s --max-time 5 "$BASE_URL" > /dev/null 2>&1; then
    echo "✅ Server is running at $BASE_URL"
else
    echo "❌ Server is not responding. Please start the server with: python run.py"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Test 1: Get Model Status"
echo "======================================================================"
curl -s "$BASE_URL/model_status" | python3 -m json.tool
echo ""

echo "======================================================================"
echo "Test 2: Get Users List"
echo "======================================================================"
curl -s "$BASE_URL/get_users" | python3 -m json.tool
echo ""

echo "======================================================================"
echo "Test 3: Get Today's Attendance"
echo "======================================================================"
curl -s "$BASE_URL/get_attendance" | python3 -m json.tool
echo ""

echo "======================================================================"
echo "Test 4: Initialize System"
echo "======================================================================"
curl -s -X POST "$BASE_URL/initialize_system" | python3 -m json.tool
echo ""

echo "======================================================================"
echo "API Testing Complete"
echo "======================================================================"
echo ""
echo "For full testing including base64 images, use Postman:"
echo "1. Import postman_collection.json"
echo "2. Set base64_image variables"
echo "3. Run the collection tests"
echo ""
echo "See POSTMAN_TESTING.md for detailed instructions"
echo ""
