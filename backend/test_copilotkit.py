#!/usr/bin/env python3
"""
Test script to verify the CopilotKit endpoint is working
"""
import asyncio
import httpx
import json

async def test_copilotkit_endpoint():
    """Test the /api/copilotkit endpoint"""
    url = "http://localhost:8000/api/copilotkit"
    
    test_request = {
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "forwardedProps": {
            "location": {
                "latitude": 37.7749,
                "longitude": -122.4194
            }
        }
    }
    
    print("Testing CopilotKit endpoint...")
    print(f"URL: {url}")
    print(f"Request: {json.dumps(test_request, indent=2)}\n")
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json=test_request,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            ) as response:
                print(f"Status: {response.status_code}")
                print(f"Headers: {dict(response.headers)}\n")
                print("Streaming response:")
                print("-" * 50)
                
                async for line in response.aiter_lines():
                    if line:
                        print(line)
                
                print("-" * 50)
                print("\n✅ Test completed successfully!")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True

async def test_health():
    """Test the health endpoint"""
    print("\n" + "=" * 50)
    print("Testing health endpoint...")
    print("=" * 50 + "\n")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def main():
    print("=" * 50)
    print("CopilotKit Backend Test Suite")
    print("=" * 50)
    
    # Test health first
    health_ok = await test_health()
    if not health_ok:
        print("\n❌ Backend is not healthy. Make sure it's running:")
        print("   cd backend && uv run uvicorn src.main:app --reload")
        return
    
    print("\n" + "=" * 50)
    print("Testing CopilotKit SSE endpoint...")
    print("=" * 50 + "\n")
    
    # Test CopilotKit endpoint
    await test_copilotkit_endpoint()

if __name__ == "__main__":
    asyncio.run(main())
