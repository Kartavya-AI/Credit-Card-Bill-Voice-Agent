import logging
from dataclasses import dataclass, field
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

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
if not outbound_trunk_id:
    raise ValueError("SIP_OUTBOUND_TRUNK_ID is not set in the environment variables.")

async def hangup_call():
    logger.info("Hanging up the call.")
    job_ctx = get_job_context()
    try:
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )
    except Exception as e:
        logger.error(f"Error hanging up call: {e}")

# Dataclass for State Management
@dataclass
class CallState:
    customer_name: str | None = None
    is_interested: bool = False
    payment_amount: str | None = None
    payment_method: str | None = None
    last_four_digits: str | None = None
    due_date: str | None = None
    objections: list[str] = field(default_factory=list)
    has_overdue_balance: bool = False

class GreetingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Emily, a payment specialist with SecureCard Financial Services.
            Your current task is to start the call about credit card bill payment.
            1. Greet the customer professionally and warmly.
            2. Confirm their name if you have it.
            3. Check if it's a good time to discuss their credit card account.
            4. Deliver your value proposition: "I'm calling to help you with convenient payment options for your credit card bill and ensure you never miss a payment."
            5. Your goal is to get a neutral or positive response to proceed to payment discussion.
            Speak at a professional but friendly pace.
            Example: "Hi, this is Emily with SecureCard Financial Services. Is this [Customer_Name]? I'm calling about your credit card account - do you have a moment to discuss some convenient payment options?"
            """
        )
    
    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Greet the customer, introduce yourself, and explain the purpose of your call.")

    @function_tool()
    async def detected_answering_machine(self, context: RunContext[CallState]):
        logger.info("Answering machine detected. Leaving a message and hanging up.")
        await context.session.generate_reply(
            instructions="""Leave a brief, professional message: "Hi, this is Emily from SecureCard Financial Services calling about your credit card payment options. Please call us back at your convenience to discuss easy payment solutions. Thank you!" After saying this, you will hang up."""
        )
        if context.session.current_speech:
            await context.session.current_speech.wait_for_playout()
        await hangup_call()

    @function_tool()
    async def proceed_to_payment_inquiry(self, context: RunContext[CallState]):
        logger.info("Handoff from Greeting to Payment Inquiry")
        return "Great! Let me help you with that.", PaymentInquiryAgent(chat_ctx=self.session._chat_ctx)

    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call. Handing off to GoodbyeAgent.")
        return "Of course. Thanks for your time.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class PaymentInquiryAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily, a payment specialist. Your current task is to inquire about their payment needs.
            Ask key questions to understand their payment situation:
            - "Are you looking to make a payment on your current balance today?"
            - "Do you have your credit card statement or know your current balance?"
            - "When is your payment due date?"
            - "Are you interested in setting up automatic payments to avoid late fees?"
            Listen carefully to their response to determine if they want to make a payment, have questions about their balance, or need payment assistance.""",
            chat_ctx=chat_ctx,
        )

    @function_tool()
    async def customer_wants_to_pay(self, context: RunContext[CallState], payment_amount: str, due_date: str = "", last_four_digits: str = ""):
        logger.info(f"Customer wants to make payment. Amount: {payment_amount}, Due: {due_date}")
        context.userdata.is_interested = True
        context.userdata.payment_amount = payment_amount
        context.userdata.due_date = due_date
        context.userdata.last_four_digits = last_four_digits
        return "Perfect! Let me help you process that payment.", PaymentProcessingAgent(chat_ctx=self.session._chat_ctx)

    @function_tool()
    async def customer_has_question(self, context: RunContext[CallState], question_type: str):
        logger.info(f"Customer has a question: {question_type}")
        return "I'd be happy to help with that.", QuestionHandlerAgent(chat_ctx=self.session._chat_ctx)

    @function_tool()
    async def customer_not_interested(self, context: RunContext[CallState]):
        logger.info("Customer is not interested in payment services.")
        return "I understand. Thank you for your time.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def customer_has_objection(self, context: RunContext[CallState], objection: str):
        logger.info(f"Customer has an objection: {objection}")
        context.userdata.objections.append(objection)
        return "I understand your concern.", ObjectionHandlerAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call. Handing off to GoodbyeAgent.")
        return "No problem. Thank you for your time.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class QuestionHandlerAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily, a payment specialist. Your task is to answer customer questions about their credit card account.
            Common questions and responses:
            - Balance inquiry: "I can help you check your current balance. Can you verify the last four digits of your card?"
            - Payment methods: "We accept bank transfers, debit cards, and online payments. Which would be most convenient for you?"
            - Due dates: "Let me help you understand your payment schedule and how to avoid late fees."
            - Auto-pay setup: "Automatic payments ensure you never miss a due date. Would you like me to explain how that works?"
            After answering their question, guide them back to making a payment or setting up payment services.""",
            chat_ctx=chat_ctx,
        )
    
    @function_tool()
    async def question_answered_proceed_to_payment(self, context: RunContext[CallState]):
        logger.info("Question answered, proceeding to payment processing.")
        return "Does that help? Now, would you like to make a payment today?", PaymentInquiryAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call. Handing off to GoodbyeAgent.")
        return "I hope that helped. Thank you for calling.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class ObjectionHandlerAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily, a payment specialist. Your task is to handle payment-related objections.
            Respond empathetically to customer concerns:
            - If "Don't have money right now": "I understand. Would you like to discuss payment plan options or a minimum payment to avoid late fees?"
            - If "Not sure about balance": "No problem. I can help you verify your current balance and payment options."
            - If "Prefer to pay online": "That's perfectly fine. Would you like me to walk you through our secure online payment portal?"
            - If "Already made payment": "Let me help verify that payment was processed correctly."
            Your goal is to resolve concerns and guide them toward a payment solution.""",
            chat_ctx=chat_ctx,
        )
    
    @function_tool()
    async def objection_resolved(self, context: RunContext[CallState]):
        logger.info("Objection resolved, returning to payment inquiry.")
        return "I'm glad we could work that out.", PaymentInquiryAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call. Handing off to GoodbyeAgent.")
        return "I understand. Please don't hesitate to call if you need assistance.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class PaymentProcessingAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""You are Emily, a payment specialist. Your task is to guide the customer through payment processing.
            For security, you'll need to:
            1. Verify their identity (last 4 digits of card, billing ZIP code)
            2. Confirm payment amount and method
            3. Process the payment securely
            4. Provide confirmation details
            Example: "To process your payment securely, I'll need to verify the last four digits of your credit card and your billing ZIP code."
            Always emphasize security and give them confirmation numbers.""",
            chat_ctx=chat_ctx,
        )
    
    @function_tool()
    async def payment_completed(self, context: RunContext[CallState], confirmation_number: str):
        logger.info(f"Payment completed with confirmation: {confirmation_number}")
        return f"Your payment has been processed successfully. Your confirmation number is {confirmation_number}.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def payment_failed(self, context: RunContext[CallState], reason: str):
        logger.info(f"Payment failed: {reason}")
        return f"I apologize, but we're having trouble processing your payment. Let me help you with an alternative method.", ObjectionHandlerAgent(chat_ctx=self.session._chat_ctx)
    
    @function_tool()
    async def end_call(self, context: RunContext[CallState]):
        logger.info("Customer requested to end call. Handing off to GoodbyeAgent.")
        return "No problem. You can always call back to complete your payment.", GoodbyeAgent(chat_ctx=self.session._chat_ctx)


class GoodbyeAgent(Agent):
    def __init__(self, chat_ctx):
        super().__init__(
            instructions="""Your task is to end the call professionally and helpfully.
            If a payment was completed, confirm it and remind them of their confirmation number.
            If no payment was made, thank them and remind them of payment options and due dates.
            Example (payment made): "Thank you for your payment today. Please keep your confirmation number for your records. Have a great day!"
            Example (no payment): "Thank you for your time. Remember, you can make payments online 24/7 or call us back anytime. Have a great day!""",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Say goodbye to the customer based on the outcome of the call."
        )
       
        if self.session.current_speech:
            await self.session.current_speech.wait_for_playout()
        await hangup_call()


async def entrypoint(ctx: JobContext):
    try:
        dial_info = json.loads(ctx.job.metadata or "{}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in job metadata: {ctx.job.metadata}, error: {e}")
        dial_info = {}
    
    phone_number = dial_info.get("phone_number")
    if not phone_number:
        logger.error("phone_number not found in metadata, shutting down.")
        return

    participant_identity = phone_number
    logger.info(f"starting outbound payment call agent for room: {ctx.room.name}, dialing: {phone_number}")
    await ctx.connect()

    logger.info("Loading VAD model...")
    vad = silero.VAD.load()
    logger.info("VAD model loaded.")

    session = AgentSession[CallState](
        llm=google.LLM(model="gemini-1.5-flash"),
        stt=deepgram.STT(model="nova-3", language="en-US"),
        tts=cartesia.TTS(model="sonic-english", voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        vad=vad,
        userdata=CallState(),
    )

    session_started = asyncio.create_task(
        session.start(
            agent=GreetingAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony()),
        )
    )

    try:
        logger.info(f"dialing sip participant: {phone_number}")
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                wait_until_answered=True, 
            )
        )
        logger.info("sip participant answered")
    except api.TwirpError as e:
        logger.error(f"error creating SIP participant: {e.message}, SIP status: {e.metadata.get('sip_status')}")
        ctx.shutdown()
        return

    await session_started
    participant = await ctx.wait_for_participant(identity=participant_identity)
    logger.info(f"participant joined: {participant.identity}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="emily-payment-specialist",
        num_idle_processes=1,
        load_threshold=float('inf'),
    ))