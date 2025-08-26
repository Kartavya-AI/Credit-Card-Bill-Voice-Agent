import asyncio
import os
import json
import time
import re
from typing import Optional
from dotenv import load_dotenv
from livekit import api

AGENT_NAME = "emily-payment-specialist"
PHONE_NUMBER_TO_CALL = "+919787264648"

REQUIRED_ENV_VARS = [
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET", 
    "LIVEKIT_URL",
    "SIP_OUTBOUND_TRUNK_ID"
]

def validate_phone_number(phone: str) -> bool:
    pattern = r'^\+[1-9]\d{1,14}$'
    return bool(re.match(pattern, phone))

def validate_environment() -> dict:
    load_dotenv(".env.local")
    env_vars = {}
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            env_vars[var] = value
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return env_vars

async def test_api_connection(env_vars: dict) -> api.LiveKitAPI:
    ws_url = env_vars["LIVEKIT_URL"]
    api_key = env_vars["LIVEKIT_API_KEY"]
    api_secret = env_vars["LIVEKIT_API_SECRET"]
    
    http_url = ws_url.replace("wss://", "https://").replace("ws://", "http://")
    
    print(f"Testing API Connection")
    print(f"LiveKit URL: {ws_url}")
    print(f"API Key: {api_key[:8]}..." if len(api_key) > 8 else f"   API Key: {api_key}")
    
    lkapi = api.LiveKitAPI(url=http_url, api_key=api_key, api_secret=api_secret)
    
    try:
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
        print(f"API connection successful. Found {len(rooms.rooms)} active rooms.")
        return lkapi
    except Exception as e:
        print(f"API connection failed: {e}")
        await lkapi.aclose()
        raise

async def test_sip_configuration(env_vars: dict) -> None:
    trunk_id = env_vars["SIP_OUTBOUND_TRUNK_ID"]
    
    print(f"\nSIP Configuration")
    print(f"Trunk ID: {trunk_id}")
    print(f"Target Phone: {PHONE_NUMBER_TO_CALL}")
    
    if not validate_phone_number(PHONE_NUMBER_TO_CALL):
        raise ValueError(f"Invalid phone number format: {PHONE_NUMBER_TO_CALL}. Must be in E.164 format (e.g., +1234567890)")
    
    print(f"Phone number format is valid (E.164)")

async def create_payment_call(lkapi: api.LiveKitAPI, trunk_id: str) -> dict:
    timestamp = int(time.time())
    room_name = f"payment-outbound-call-{timestamp}"
    
    print(f"\n🏦 Creating Credit Card Payment Call")
    print(f"   Room Name: {room_name}")
    print(f"   Agent: Emily - Payment Specialist")
    print(f"   Company: SecureCard Financial Services")
    print(f"   Phone: {PHONE_NUMBER_TO_CALL}")
    
    metadata = json.dumps({
        "phone_number": PHONE_NUMBER_TO_CALL,
        "call_type": "credit_card_payment",
        "company": "SecureCard Financial Services",
        "agent_name": "Emily",
        "created_at": timestamp,
        "purpose": "payment_collection"
    })
    print(f"📋 Call Metadata: {json.loads(metadata)}")
    
    try:
        print(f"Dispatching payment agent...")
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=metadata
            )
        )
        call_info = {
            "dispatch_id": dispatch.id,
            "room": dispatch.room,
            "agent_name": AGENT_NAME,
            "phone_number": PHONE_NUMBER_TO_CALL,
            "created_at": timestamp
        }
        
        print(f"Payment call dispatch created successfully!")
        print(f"Dispatch ID: {dispatch.id}")
        print(f"Room: {dispatch.room}")
        print(f"Status: Active")
        
        return call_info
        
    except api.TwirpError as e:
        print(f"\nLiveKit API Error:")
        print(f"   Code: {e.code}")
        print(f"   Message: {e.message}")
        
        if "object cannot be found" in e.message.lower():
            print(f"\n🔧 Troubleshooting Tips:")
            print(f"   1. Verify SIP trunk exists: {trunk_id}")
            print(f"   2. Check trunk is configured for outbound calls")
            print(f"   3. Confirm SIP provider credentials are correct")
            print(f"   4. Ensure trunk has sufficient balance/credits")
            
        elif "agent" in e.message.lower():
            print(f"\n🔧 Agent Troubleshooting:")
            print(f"   1. Verify agent name: {AGENT_NAME}")
            print(f"   2. Check agent is deployed and running")
            print(f"   3. Confirm agent has proper permissions")
            
        elif "unauthorized" in e.message.lower():
            print(f"\n🔧 Authentication Issues:")
            print(f"   1. Check API key and secret are correct")
            print(f"   2. Verify API key has proper permissions")
            print(f"   3. Ensure LiveKit URL is correct")
            
        raise

async def monitor_call_status(lkapi: api.LiveKitAPI, room_name: str, duration: int = 60) -> None:
    """Monitor the call status for a specified duration."""
    print(f"\n👀 Monitoring call status for {duration} seconds...")
    
    start_time = time.time()
    last_participant_count = 0
    
    while time.time() - start_time < duration:
        try:
            # Get room information
            room_info = await lkapi.room.list_participants(
                api.ListParticipantsRequest(room=room_name)
            )
            
            participant_count = len(room_info.participants)
            
            # Only print updates when participant count changes
            if participant_count != last_participant_count:
                print(f"   📊 Room: {room_name}")
                print(f"   👥 Participants: {participant_count}")
                
                for participant in room_info.participants:
                    status = "🟢 Connected" if participant.state == 0 else "🔴 Disconnected"
                    print(f"      - {participant.identity}: {status}")
                
                last_participant_count = participant_count
            
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except api.TwirpError as e:
            if "room not found" in e.message.lower():
                print(f"   ℹ️  Call completed - room closed")
                break
            else:
                print(f"   ⚠️  Monitoring error: {e.message}")
        
        except Exception as e:
            print(f"   ⚠️  Unexpected monitoring error: {e}")

async def cleanup_old_rooms(lkapi: api.LiveKitAPI, max_age_minutes: int = 30) -> None:
    """Clean up old payment call rooms."""
    try:
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
        current_time = time.time()
        
        old_rooms = []
        for room in rooms.rooms:
            if room.name.startswith("payment-outbound-call-"):
                # Extract timestamp from room name
                try:
                    timestamp_str = room.name.split("-")[-1]
                    room_timestamp = int(timestamp_str)
                    age_minutes = (current_time - room_timestamp) / 60
                    
                    if age_minutes > max_age_minutes:
                        old_rooms.append(room.name)
                except (ValueError, IndexError):
                    continue
        
        if old_rooms:
            print(f"\n🧹 Cleaning up {len(old_rooms)} old rooms...")
            for room_name in old_rooms:
                try:
                    await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
                    print(f"   ✅ Deleted: {room_name}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {room_name}: {e}")
        else:
            print(f"\n✨ No old rooms to clean up")
            
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")

async def main():
    """Main function to orchestrate the payment call."""
    print("🎯 SecureCard Financial Services - Payment Collection Call")
    print("=" * 60)
    
    try:
        # Step 1: Validate environment
        print("1️⃣ Validating environment configuration...")
        env_vars = validate_environment()
        print("✅ Environment validation successful")
        
        # Step 2: Test API connection
        print("\n2️⃣ Testing LiveKit API connection...")
        lkapi = test_api_connection(env_vars)
        
        # Step 3: Test SIP configuration
        print("3️⃣ Validating SIP configuration...")
        await test_sip_configuration(env_vars)
        
        # Step 4: Clean up old rooms
        print("\n4️⃣ Cleaning up old rooms...")
        await cleanup_old_rooms(lkapi)
        
        # Step 5: Create the payment call
        print("\n5️⃣ Creating payment collection call...")
        call_info = await create_payment_call(lkapi, env_vars["SIP_OUTBOUND_TRUNK_ID"])
        
        # Step 6: Monitor the call
        print("\n6️⃣ Call initiated successfully!")
        print(f"   📞 Emily is now calling: {PHONE_NUMBER_TO_CALL}")
        print(f"   💳 Purpose: Credit card payment collection")
        print(f"   🏢 Company: SecureCard Financial Services")
        
        # Optional: Monitor call status
        monitor_choice = input("\n❓ Would you like to monitor the call status? (y/N): ").lower()
        if monitor_choice in ['y', 'yes']:
            await monitor_call_status(lkapi, call_info["room"], duration=120)
        
        print(f"\n🎉 Call dispatch completed successfully!")
        print(f"   Dispatch ID: {call_info['dispatch_id']}")
        print(f"   Room: {call_info['room']}")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nMake sure your .env.local file contains all required variables:")
        for var in REQUIRED_ENV_VARS:
            print(f"   {var}=your_value_here")
            
    except api.TwirpError as e:
        print(f"LiveKit API Error: {e.message}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    finally:
        if 'lkapi' in locals():
            await lkapi.aclose()
            print(f"\nAPI connection closed")

if __name__ == "__main__":
    asyncio.run(main())