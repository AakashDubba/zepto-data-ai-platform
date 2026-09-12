"""
Zepto Support Assistant — API Test Script
==========================================
Tests the FastAPI /ask endpoint with policy and general queries.
Validates JSON responses and logs raw outputs.
"""

import requests
import json
import time
import sys
import subprocess
import os

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:7860"


def test_api():
    """Test the /ask endpoint with both policy and general queries."""
    print("=" * 60)
    print("  Zepto Support Assistant — API Tests")
    print("=" * 60)

    # Wait for server to be ready
    print("\n  Waiting for server to be ready...")
    for i in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/", timeout=2)
            if resp.status_code == 200:
                print(f"  Server is ready! (attempt {i+1})")
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        print("  ❌ Server did not start within 30 seconds")
        sys.exit(1)

    # ── Test 1: Policy Query ──
    print("\n" + "─" * 60)
    print("  TEST 1: Policy Query")
    print("─" * 60)
    
    policy_query = {"query": "What is Zepto's delivery policy?"}
    print(f"  Request: {json.dumps(policy_query)}")
    
    resp = requests.post(f"{BASE_URL}/ask", json=policy_query, timeout=30)
    print(f"  Status: {resp.status_code}")
    result = resp.json()
    print(f"  Raw Response: {json.dumps(result, indent=2)}")
    
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "answer" in result, "Missing 'answer' field"
    assert "sources" in result, "Missing 'sources' field"
    assert "confidence" in result, "Missing 'confidence' field"
    assert len(result["sources"]) > 0, "Expected non-empty sources for policy query"
    assert isinstance(result["confidence"], (int, float)), "Confidence must be numeric"
    print(f"  ✅ Policy query test PASSED")
    print(f"     Sources: {result['sources']}")
    print(f"     Confidence: {result['confidence']}")

    # ── Test 2: General (Out-of-Scope) Query ──
    print("\n" + "─" * 60)
    print("  TEST 2: General (Out-of-Scope) Query")
    print("─" * 60)
    
    general_query = {"query": "What is the capital of France?"}
    print(f"  Request: {json.dumps(general_query)}")
    
    resp = requests.post(f"{BASE_URL}/ask", json=general_query, timeout=30)
    print(f"  Status: {resp.status_code}")
    result = resp.json()
    print(f"  Raw Response: {json.dumps(result, indent=2)}")
    
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "answer" in result, "Missing 'answer' field"
    assert "sources" in result, "Missing 'sources' field"
    assert result["sources"] == [], "Expected empty sources for general query"
    assert "Zepto policies" in result["answer"], "Expected canned out-of-scope response"
    print(f"  ✅ General query test PASSED")
    print(f"     Sources: {result['sources']} (empty as expected)")

    # ── Test 3: Another Policy Query (refund) ──
    print("\n" + "─" * 60)
    print("  TEST 3: Policy Query (Refund)")
    print("─" * 60)
    
    refund_query = {"query": "How do I get a refund on my order?"}
    print(f"  Request: {json.dumps(refund_query)}")
    
    resp = requests.post(f"{BASE_URL}/ask", json=refund_query, timeout=30)
    print(f"  Status: {resp.status_code}")
    result = resp.json()
    print(f"  Raw Response: {json.dumps(result, indent=2)}")
    
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(result["sources"]) > 0, "Expected non-empty sources for refund query"
    print(f"  ✅ Refund query test PASSED")

    print("\n" + "=" * 60)
    print("  ALL API TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
