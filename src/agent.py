import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import os
import json
import asyncio

from livekit import rtc, api
from livekit.agents import (Agent, AgentSession, JobContext, RoomInputOptions, RunContext, WorkerOptions, cli, get_job_context, function_tool)
from livekit.plugins import cartesia, deepgram, google, noise_cancellation, silero

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emily-agent")

load_dotenv(".env.local")

REQUIRED_ENV_VARS = [
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET", 
    "LIVEKIT_URL",
    "SIP_OUTBOUND_TRUNK_ID"
]

def validate_environment():
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.info("All required environment variables validated successfully")

validate_environment()
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

def validate_phone_number(phone: str) -> bool:
    pattern = r'^\+[1-9]\d{1,14}$'
    return bool(re.match(pattern, phone))

def validate_payment_amount(amount: str) -> bool:
    try:
        float_amount = float(amount.replace('$', '').replace(',', ''))
        return 0 < float_amount <= 50000
    except (ValueError, AttributeError):
        return False

def sanitize_log_data(data: str) -> str:
    patterns = [
        (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '****-****-****-****'),
        (r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****'),
    ]
    
    for pattern, replacement in patterns:
        data = re.sub(pattern, replacement, data)
    return data

async def hangup_call_with_retry(max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            logger.info(f"Hanging up the call (attempt {attempt + 1})")
            job_ctx = get_job_context()
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )
            logger.info("Call hung up successfully")
            return
        except Exception as e:
            logger.error(f"Error hanging up call (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error("Failed to hang up call after all retries")

@dataclass
class CallState:
    customer_name: Optional[str] = None
    phone_number: Optional[str] = None
    is_interested: bool = False
    payment_amount: Optional[str] = None
    payment_method: Optional[str] = None
    last_four_digits: Optional[str] = None
    billing_zip: Optional[str] = None
    due_date: Optional[str] = None
    current_balance: Optional[str] = None
    objections: list[str] = field(default_factory=list)
    has_overdue_balance: bool = False
    call_start_time: float = field(default_factory=time.time)
    interaction_count: int = 0
    payment_confirmed: bool = False
    confirmation_number: Optional[str] = None

class GreetingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Emily, a professional payment specialist with SecureCard Financial Services.
            Your current task is to start the call about credit card bill payment.
            
            IMPORTANT GUIDELINES:
            1. Greet the customer professionally and warmly
            2. Introduce yourself clearly: "This is Emily with SecureCard Financial Services"
            3. Ask for confirmation of their name politely
            4. Check if it's a good time to discuss their credit card account
            5. Explain the purpose: "I'm calling to help you with convenient payment options for your credit card bill"
            6. Listen carefully for verbal cues that indicate an answering machine
            7. If you detect hesitation or disinterest, be respectful and offer to call back
            
            Keep your tone professional, friendly, and respectful. Speak clearly and at a moderate pace.
            
            Example opening: "Hello, this is Emily calling from SecureCard Financial Services. May I please speak with [Customer Name]? I'm calling today to help you with convenient payment options for your credit card account. Is this a good time to chat for just a few minutes?"
            """
        )
    
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the customer professionally, introduce yourself as Emily from SecureCard Financial Services, and explain you're calling about payment options for their credit card account."
        )

    @function_tool()
    async def detected_answering_machine(self, context: RunContext[CallState]):
        logger.info("Answering machine detected. Leaving message and hanging up.")
        await context.session.generate_reply(
            instructions="""Leave a brief, professional message: "Hello, this is Emily from SecureCard Financial Services. I'm calling about convenient payment options for your credit card account. Please call us back at 1-800-555-0123 at your convenience to discuss easy payment solutions that can help you avoid late fees. Thank you and have a great day!" After delivering this message completely, you will end the call."""
        )
        if context.session.current_speech:
            await context.session.current_speech.wait_for_playout()
        await hangup_call_with_retry()

    @function_tool()
    async def customer_confirmed_identity(self, context: RunContext[CallState], customer_name: str):
        context.userdata.customer_name = customer_name
        context.userdata.interaction_count += 1
        logger.info(f"Customer identity confirmed: {sanitize_log_data(customer_name)}")

    @function_tool()
    async def proceed_to_payment_inquiry(self, context: RunContext[CallState]):
        logger.info("Transitioning from Greeting to Payment Inquiry")
        context.userdata.interaction_count += 1
        return "Wonderful! Let me help you with that.", PaymentInquiryAgent(chat_ctx=self.session._chat_ctx)

    @function_tool()
    async def customer_requests_callback(self, context: RunContext[CallState], preferred_time: str = ""):
        logger.info(f"Customer requested callback for: {preferred_time}")
        await context.session.generate_reply(
            instructions=f"Acknowledge their request professionally: 'Absolutely! I'll make sure someone calls you back {preferred_time if preferred_time else 'at a more convenient time'}. Thank you for your time, and have a great day!'"
        )
        if context.session.current_speech:
            await context.session.current_speech.wait_for_playout()
        await hangup_call_with_retry()

    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call during greeting.")
        return "Of course, I understand. Thank you for your time.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class PaymentInquiryAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily, a payment specialist. Your task is to understand their payment needs and situation.
            
            Key questions to ask (naturally, not as a rigid script):
            1. "Are you looking to make a payment on your current balance today?"
            2. "Do you have your credit card statement handy, or would you like me to help you with your current balance?"
            3. "When is your next payment due date?"
            4. "What payment amount were you thinking of making?"
            5. "Are you interested in learning about automatic payments to help avoid late fees in the future?"
            
            IMPORTANT:
            - Listen carefully to their responses
            - Be empathetic if they mention financial difficulties
            - Offer solutions that match their needs
            - Don't be pushy - focus on being helpful
            - If they seem confused about their balance, offer to help clarify
            
            Your goal is to understand their situation and guide them toward a payment solution that works for them.""",
            chat_ctx=chat_ctx,
        )

    @function_tool()
    async def customer_wants_to_pay(self, context: RunContext[CallState], payment_amount: str, due_date: str = "", current_balance: str = ""):
        if not validate_payment_amount(payment_amount):
            logger.warning(f"Invalid payment amount provided: {payment_amount}")
            await context.session.generate_reply(
                instructions="The payment amount seems unclear. Could you please confirm the amount you'd like to pay? For example, '$150' or '$75.50'?"
            )
            return
        
        logger.info(f"Customer wants to make payment. Amount: {payment_amount}")
        context.userdata.is_interested = True
        context.userdata.payment_amount = payment_amount
        context.userdata.due_date = due_date
        context.userdata.current_balance = current_balance
        context.userdata.interaction_count += 1
        return "Perfect! Let me help you process that payment securely.", PaymentProcessingAgent(chat_ctx=self.session._chat_ctx)

    @function_tool()
    async def customer_has_question(self, context: RunContext[CallState], question_type: str):
        logger.info(f"Customer has question: {sanitize_log_data(question_type)}")
        context.userdata.interaction_count += 1
        return "I'd be happy to help you with that.", QuestionHandlerAgent(chat_ctx=self.session._chat_ctx)

    @function_tool()
    async def customer_not_interested(self, context: RunContext[CallState], reason: str = ""):
        logger.info(f"Customer not interested. Reason: {sanitize_log_data(reason)}")
        context.userdata.interaction_count += 1
        return "I completely understand. Thank you for your time.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def customer_has_objection(self, context: RunContext[CallState], objection: str):
        logger.info(f"Customer objection: {sanitize_log_data(objection)}")
        context.userdata.objections.append(objection)
        context.userdata.interaction_count += 1
        return "I understand your concern. Let me help address that.", ObjectionHandlerAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def customer_needs_balance_info(self, context: RunContext[CallState]):
        logger.info("Customer needs balance information")
        context.userdata.interaction_count += 1
        return "Let me help you with your balance information.", QuestionHandlerAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call during payment inquiry.")
        return "No problem at all. Thank you for your time.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class QuestionHandlerAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily, a knowledgeable payment specialist. Answer customer questions helpfully and accurately.
            
            Common questions and professional responses:
            - Balance inquiry: "I can help you check your current balance. For security, I'll need to verify the last four digits of your card and your billing ZIP code."
            - Payment methods: "We accept several convenient options: bank transfers, debit cards, and secure online payments. Which would work best for you?"
            - Due dates: "Let me help you understand your payment schedule and share some tips to avoid late fees."
            - Late fees: "I understand your concern about late fees. Let me explain how we can help prevent them in the future."
            - Auto-pay: "Automatic payments are a great way to ensure you never miss a due date. Would you like me to explain how our auto-pay service works?"
            - Minimum payment: "I can help you understand your minimum payment options and what works best for your situation."
            
            After answering their question thoroughly:
            - Ask if they have any other questions
            - Naturally guide back to helping with a payment
            - Be patient and thorough - good customer service builds trust""",
            chat_ctx=chat_ctx,
        )
    
    @function_tool()
    async def question_answered_proceed_to_payment(self, context: RunContext[CallState]):
        logger.info("Question answered, proceeding to payment inquiry.")
        context.userdata.interaction_count += 1
        return "I hope that helps! Now, would you like to take care of a payment today?", PaymentInquiryAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def customer_has_more_questions(self, context: RunContext[CallState]):
        logger.info("Customer has additional questions")
        context.userdata.interaction_count += 1
        await context.session.generate_reply(
            instructions="Encourage them to ask: 'Of course! What other questions can I help you with?'"
        )
    
    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call after questions.")
        return "I'm glad I could help answer your questions. Thank you for calling.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class ObjectionHandlerAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily, an empathetic payment specialist. Handle objections with understanding and offer practical solutions.
            
            Professional responses to common objections:
            - "Don't have money right now": "I completely understand financial situations can be tight. Would you be interested in discussing a payment plan or making a smaller payment today to help with your account standing?"
            - "Not sure about balance": "That's perfectly fine. Let me help verify your current balance and payment options so you have all the information you need."
            - "Prefer to pay online": "That's absolutely fine! Many customers prefer online payments. Would you like me to walk you through our secure online portal, or would you prefer I send you the link?"
            - "Already made payment": "Thank you for letting me know. Let me help verify that your payment was processed correctly and update your account."
            - "Don't trust phone payments": "I completely understand that concern. Security is very important. Let me explain our security measures, or I can help you with other secure payment options."
            - "Too busy right now": "I understand you're busy. Would you prefer if I called back at a better time, or would you like me to quickly share some convenient payment options?"
            
            Always:
            - Acknowledge their concern first
            - Show empathy and understanding  
            - Offer practical alternatives
            - Don't argue or pressure
            - Focus on solving their problem""",
            chat_ctx=chat_ctx,
        )
    
    @function_tool()
    async def objection_resolved(self, context: RunContext[CallState]):
        logger.info("Objection resolved, returning to payment inquiry.")
        context.userdata.interaction_count += 1
        return "I'm glad we could work that out. Now, how can I best help you with your payment today?", PaymentInquiryAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def offer_alternative_solution(self, context: RunContext[CallState], solution_type: str):
        logger.info(f"Offering alternative solution: {solution_type}")
        context.userdata.interaction_count += 1
        
        solutions = {
            "online_payment": "I can send you a secure link to make your payment online at your convenience.",
            "callback": "I can arrange for someone to call you back at a better time.",
            "payment_plan": "We can discuss setting up a payment plan that works with your budget.",
            "email_info": "I can email you all the payment information so you can handle it when convenient."
        }
        
        return solutions.get(solution_type, "Let me see what other options I can offer you."), PaymentInquiryAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer maintained objection and requested to end call.")
        return "I understand completely. Please don't hesitate to call us if you need any assistance in the future.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class PaymentProcessingAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily, a payment specialist handling secure payment processing. 
            
            Security verification process:
            1. "For your security, I need to verify a few details before processing your payment."
            2. "Can you please provide the last four digits of your credit card?"
            3. "And can you confirm your billing ZIP code?"
            4. "Perfect! Now let me confirm the payment amount: $[amount]. Is that correct?"
            5. "Which payment method would you prefer: bank transfer, debit card, or online payment?"
            
            Payment processing steps:
            - Verify all security information
            - Confirm payment amount and method
            - Generate confirmation number (format: SC[YYYYMMDD][####])
            - Provide confirmation details
            - Explain when payment will post
            
            Always emphasize:
            - Security and protection of their information
            - Confirmation details they should keep
            - When the payment will appear on their account
            - How to contact us if they have questions
            
            Example: "Your payment of $150.00 has been processed successfully. Your confirmation number is SC20240115001. Please keep this number for your records. The payment will post to your account within 1-2 business days.""",
            chat_ctx=chat_ctx,
        )
    
    @function_tool()
    async def verify_customer_info(self, context: RunContext[CallState], last_four_digits: str, billing_zip: str):
        if not re.match(r'^\d{4}$', last_four_digits):
            await context.session.generate_reply(
                instructions="I need exactly four digits. Could you please provide just the last four digits of your credit card?"
            )
            return
        
        if not re.match(r'^\d{5}$', billing_zip):
            await context.session.generate_reply(
                instructions="I need your 5-digit ZIP code. Could you please provide your billing ZIP code?"
            )
            return
        
        context.userdata.last_four_digits = last_four_digits
        context.userdata.billing_zip = billing_zip
        context.userdata.interaction_count += 1
        logger.info(f"Customer info verified: ****{last_four_digits}, ZIP: {billing_zip}")
    
    @function_tool()
    async def process_payment(self, context: RunContext[CallState], payment_method: str):
        if not context.userdata.last_four_digits or not context.userdata.billing_zip:
            await context.session.generate_reply(
                instructions="I still need to verify your information before processing the payment. Could you provide the last four digits of your card and billing ZIP code?"
            )
            return
    
        confirmation_number = f"SC{time.strftime('%Y%m%d')}{str(int(time.time()))[-4:]}"
        context.userdata.payment_method = payment_method
        context.userdata.payment_confirmed = True
        context.userdata.confirmation_number = confirmation_number
        context.userdata.interaction_count += 1
        
        logger.info(f"Payment processed: {context.userdata.payment_amount} via {payment_method}, Confirmation: {confirmation_number}")
        return f"Excellent! Your payment of {context.userdata.payment_amount} has been processed successfully via {payment_method}. Your confirmation number is {confirmation_number}.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def payment_failed(self, context: RunContext[CallState], reason: str):
        logger.warning(f"Payment failed: {sanitize_log_data(reason)}")
        context.userdata.interaction_count += 1
        return f"I apologize, but we're experiencing a technical issue with processing your payment. Let me help you with an alternative method.", ObjectionHandlerAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def customer_wants_different_amount(self, context: RunContext[CallState], new_amount: str):
        if not validate_payment_amount(new_amount):
            await context.session.generate_reply(
                instructions="Could you please clarify the payment amount? For example, '$150' or '$75.50'?"
            )
            return
        
        context.userdata.payment_amount = new_amount
        context.userdata.interaction_count += 1
        logger.info(f"Payment amount changed to: {new_amount}")
        await context.session.generate_reply(
            instructions=f"Got it! I've updated your payment amount to {new_amount}. Let me continue with the security verification."
        )
    
    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call during payment processing.")
        return "No problem. You can always call back to complete your payment when you're ready.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class GoodbyeAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily. Your task is to end the call professionally based on what happened during the call.
            
            If a payment was successfully completed:
            - Thank them for the payment
            - Remind them of their confirmation number 
            - Tell them when the payment will post
            - Provide customer service number for questions
            Example: "Thank you for your payment of $150 today. Your confirmation number is SC20240115001 - please keep this for your records. The payment will post to your account in 1-2 business days. If you have any questions, you can reach us at 1-800-555-0123. Have a wonderful day!"
            
            If no payment was made:
            - Thank them for their time
            - Remind them of payment options and due dates if discussed
            - Provide helpful contact information
            - Leave the door open for future contact
            Example: "Thank you for taking the time to speak with me today. Remember, you can make payments online 24/7 at our secure portal, or call us back anytime at 1-800-555-0123. We're here to help make payments convenient for you. Have a great day!"
            
            Always end on a positive, helpful note that reinforces good customer service.""",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Provide a professional goodbye based on the outcome of the call. If a payment was made, reference the confirmation number and next steps. If no payment was made, thank them and provide helpful contact information."
        )
       
        if self.session.current_speech:
            await self.session.current_speech.wait_for_playout()
        
        await asyncio.sleep(2)
        await hangup_call_with_retry()


async def create_sip_participant_with_retry(ctx: JobContext, phone_number: str, max_retries: int = 3):
    participant_identity = phone_number
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Dialing SIP participant: {phone_number} (attempt {attempt + 1})")
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=outbound_trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=participant_identity,
                    wait_until_answered=True,
                )
            )
            logger.info("SIP participant answered successfully")
            return participant_identity
            
        except api.TwirpError as e:
            logger.error(f"Error creating SIP participant (attempt {attempt + 1}): {e.message}")
            logger.error(f"SIP status: {e.metadata.get('sip_status')}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("Failed to create SIP participant after all retries")
                raise


async def entrypoint(ctx: JobContext):
    try:
        try:
            dial_info = json.loads(ctx.job.metadata or "{}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in job metadata: {ctx.job.metadata}, error: {e}")
            return

        phone_number = dial_info.get("phone_number")
        if not phone_number:
            logger.error("phone_number not found in metadata")
            return

        if not validate_phone_number(phone_number):
            logger.error(f"Invalid phone number format: {phone_number}")
            return

        logger.info(f"Starting outbound payment call for room: {ctx.room.name}, dialing: {sanitize_log_data(phone_number)}")

        await ctx.connect()
        logger.info("Loading VAD model...")
        try:
            vad = silero.VAD.load()
            logger.info("VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            return

        session = AgentSession[CallState](
            llm=google.LLM(model="gemini-1.5-flash"),
            stt=deepgram.STT(model="nova-3", language="en-US"),
            tts=cartesia.TTS(model="sonic-english", voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
            vad=vad,
            userdata=CallState(phone_number=phone_number),
        )

        session_started = asyncio.create_task(
            session.start(
                agent=GreetingAgent(),
                room=ctx.room,
                room_input_options=RoomInputOptions(
                    noise_cancellation=noise_cancellation.BVCTelephony()
                ),
            )
        )

        try:
            participant_identity = await create_sip_participant_with_retry(ctx, phone_number)
        except api.TwirpError:
            ctx.shutdown()
            return

        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"Participant joined: {participant.identity}")
        call_duration = time.time() - session.userdata.call_start_time
        logger.info(f"Call completed. Duration: {call_duration:.1f}s, Interactions: {session.userdata.interaction_count}, Payment: {session.userdata.payment_confirmed}")

    except Exception as e:
        logger.error(f"Unexpected error in entrypoint: {e}")
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="emily-payment-specialist",
        num_idle_processes=1,
        load_threshold=float('inf'),
    ))