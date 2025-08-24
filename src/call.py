import asyncio
import os
import json
import time
from dotenv import load_dotenv
from livekit import api

AGENT_NAME = "jack-outbound-caller"
PHONE_NUMBER_TO_CALL = "+919787264648"

async def test_sip_configuration():
    load_dotenv(".env.local")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    ws_url = os.getenv("LIVEKIT_URL")
    trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

    print("Configuration Check")
    print(f"LiveKit URL: {ws_url}")
    print(f"API Key: {api_key[:8]}..." if api_key else "API Key: NOT SET")
    print(f"Trunk ID: {trunk_id}")
    
    if not all([api_key, api_secret, ws_url, trunk_id]):
        missing = []
        if not api_key: missing.append("LIVEKIT_API_KEY")
        if not api_secret: missing.append("LIVEKIT_API_SECRET")
        if not ws_url: missing.append("LIVEKIT_URL")
        if not trunk_id: missing.append("SIP_OUTBOUND_TRUNK_ID")
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    http_url = ws_url.replace("wss://", "https://").replace("ws://", "http://")
    lkapi = api.LiveKitAPI(url=http_url, api_key=api_key, api_secret=api_secret)

    try:
        print("\nTesting API Connection")
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
        print(f"API connection successful. Found {len(rooms.rooms)} rooms.")
        return lkapi, trunk_id
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        await lkapi.aclose()
        raise

async def make_call():
    try:
        lkapi, trunk_id = await test_sip_configuration()
        room_name = f"hotel-outbound-call-{int(time.time())}"
        print(f"\nCreating Call")
        print(f"Room name: {room_name}")
        print(f"Phone number: {PHONE_NUMBER_TO_CALL}")
        
        metadata = json.dumps({
            "phone_number": PHONE_NUMBER_TO_CALL
        })
        
        print("Creating agent dispatch...")
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=metadata
            )
        )
        print(f"Dispatch created successfully!")
        print(f"Dispatch ID: {dispatch.id}")
        print(f"Room: {dispatch.room}")
        
    except api.TwirpError as e:
        print(f"\nLiveKit API Error:")
        print(f"Code: {e.code}")
        print(f"Message: {e.message}")
        print(f"Meta: {e.metadata}")
        
        if "object cannot be found" in e.message.lower():
            print(f"\nTroubleshooting Tips:")
            print(f"1. Check your SIP_OUTBOUND_TRUNK_ID: {os.getenv('SIP_OUTBOUND_TRUNK_ID')}")
            print(f"2. Verify the trunk exists in your LiveKit dashboard")
            print(f"3. Ensure the trunk is configured for outbound calls")
            print(f"4. Check that your SIP provider credentials are correct")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    finally:
        if 'lkapi' in locals():
            await lkapi.aclose()

if __name__ == "__main__":
    asyncio.run(make_call())