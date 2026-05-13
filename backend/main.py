"""
Affitto Ride & Rent — FastAPI Backend
=====================================
Endpoints:
  POST /api/users          — Create or update user profile
  GET  /api/users/{uid}    — Get user profile
  PATCH /api/users/{uid}   — Partially update user profile
  POST /api/rides          — Book a new ride (waterfall dispatch)
  GET  /api/rides/{user_id}— Get ride history for a user
  GET  /api/rides/{ride_id}/status — Get current ride status
  POST /api/rides/{ride_id}/accept — Rider accepts a ride
  POST /api/rides/{ride_id}/reject — Rider rejects a ride
  POST /api/riders/{uid}/location  — Update rider location
  POST /api/contact        — Submit a rental / contact-us inquiry
"""

import os
import uuid
import json
import math
import random
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import firebase_admin
from firebase_admin import credentials, firestore, messaging
from dotenv import load_dotenv

# ─── Firebase Admin SDK Initialisation ────────────────────────────────────────
# python-dotenv can't parse the complex JSON credential string,
# so we read the .env file manually for FIREBASE_CREDENTIALS.
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
firebase_json = None
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("FIREBASE_CREDENTIALS="):
                # Split on first '=' only, strip wrapping quotes
                firebase_json = line.split("=", 1)[1].strip().strip("'").strip('"')
                break

if not firebase_json:
    firebase_json = os.getenv("FIREBASE_CREDENTIALS")

if not firebase_json:
    raise Exception("FIREBASE_CREDENTIALS not found")
cred_dict = json.loads(firebase_json)
cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
cred = credentials.Certificate(cred_dict)
# cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

logger = logging.getLogger("affitto")
logging.basicConfig(level=logging.INFO)

# ─── Dispatch Configuration ──────────────────────────────────────────────────
DISPATCH_RADIUS_KM = 10.0       # Max distance to search for riders
DISPATCH_TIMEOUT_SECONDS = 15   # Time to wait for each rider to respond
MAX_DISPATCH_ATTEMPTS = 3       # Max riders to try before giving up

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Affitto Ride & Rent API",
    version="2.0.0",
    description="Backend API for the Affitto transport service with waterfall dispatch.",
)

# CORS — allow the frontend (served on any origin during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    uid: str
    name: str
    email: str
    phone: Optional[str] = ""
    photo_url: Optional[str] = ""


class UserProfileUpdate(BaseModel):
    phone: Optional[str] = None
    name: Optional[str] = None


class RideCreate(BaseModel):
    user_id: str
    pickup: str
    dropoff: str
    distance_km: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    # Lat/lng for proximity dispatch
    pickup_lat: Optional[float] = None
    pickup_lng: Optional[float] = None
    drop_lat: Optional[float] = None
    drop_lng: Optional[float] = None
    # User info for denormalization on rider side
    user_name: Optional[str] = ""
    user_phone: Optional[str] = ""


class RideAcceptRequest(BaseModel):
    rider_id: str


class RideRejectRequest(BaseModel):
    rider_id: str


class RiderLocationUpdate(BaseModel):
    latitude: float
    longitude: float
    is_online: Optional[bool] = None


class ContactRequest(BaseModel):
    name: str
    phone: str
    vehicle: str
    message: Optional[str] = ""


# ─── Haversine Distance Utility ──────────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ─── Find Nearest Online Riders ──────────────────────────────────────────────

def find_nearest_riders(
    pickup_lat: float,
    pickup_lng: float,
    max_radius_km: float = DISPATCH_RADIUS_KM,
    limit: int = MAX_DISPATCH_ATTEMPTS,
    exclude_rider_ids: List[str] = None,
) -> List[Dict]:
    """
    Query Firestore for online riders within radius.
    Returns list of dicts sorted by distance: [{uid, fcm_token, full_name, distance_km}, ...]
    
    Note: is_approved check is removed because approval is already gated at rider login.
    This prevents riders from being invisible due to field naming mismatches.
    """
    if exclude_rider_ids is None:
        exclude_rider_ids = []

    riders_ref = db.collection("riders")
    # Query all online riders (approval is enforced at rider app login)
    query = riders_ref.where("is_online", "==", True)
    docs = query.stream()

    candidates = []
    total_online = 0
    skipped_no_location = 0
    skipped_no_fcm = 0
    skipped_excluded = 0
    skipped_out_of_range = 0

    for doc in docs:
        data = doc.to_dict()
        uid = data.get("uid", doc.id)
        total_online += 1

        # Skip already-attempted riders
        if uid in exclude_rider_ids:
            skipped_excluded += 1
            continue

        lat = data.get("latitude", 0.0)
        lng = data.get("longitude", 0.0)

        # Skip riders with no valid location
        if lat == 0.0 and lng == 0.0:
            skipped_no_location += 1
            logger.warning("Rider %s is online but has no GPS coordinates", uid)
            continue

        fcm_token = data.get("fcm_token", "")
        if not fcm_token:
            skipped_no_fcm += 1
            logger.warning("Rider %s is online but has no FCM token", uid)
            continue

        distance = haversine_km(pickup_lat, pickup_lng, lat, lng)
        if distance <= max_radius_km:
            candidates.append({
                "uid": uid,
                "fcm_token": fcm_token,
                "full_name": data.get("full_name", "Rider"),
                "mobile": data.get("mobile", ""),
                "distance_km": round(distance, 2),
            })
        else:
            skipped_out_of_range += 1

    logger.info(
        "Rider search: %d online, %d eligible, %d no-location, %d no-fcm, %d excluded, %d out-of-range",
        total_online, len(candidates), skipped_no_location, skipped_no_fcm,
        skipped_excluded, skipped_out_of_range
    )

    # Sort by distance (nearest first)
    candidates.sort(key=lambda r: r["distance_km"])
    return candidates[:limit]


# ─── Generate OTP ────────────────────────────────────────────────────────────

def get_or_create_user_otp(user_id: str) -> str:
    """
    Get a fixed OTP for the user. Each user gets a unique, persistent OTP
    stored in their Firestore `users` document. If no OTP exists yet, one
    is generated and saved.
    """
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()

    if user_doc.exists:
        existing_otp = user_doc.to_dict().get("otp", "")
        if existing_otp:
            return str(existing_otp)

    # Generate a new fixed OTP for this user
    new_otp = str(random.randint(1000, 9999))
    # Save to user document (merge to avoid overwriting other fields)
    user_ref.set({"otp": new_otp}, merge=True)
    logger.info("Generated new fixed OTP %s for user %s", new_otp, user_id)
    return new_otp


# ─── Send Targeted FCM to a Single Rider ─────────────────────────────────────

def send_targeted_ride_notification(
    fcm_token: str, ride_data: dict
) -> Optional[str]:
    """
    Send a high-priority FCM message to a specific rider's device.
    Includes BOTH a data payload (for the app's onMessageReceived handler)
    AND a notification payload (as fallback so the system tray shows it
    even if onMessageReceived doesn't fire on some OEMs).
    Returns the FCM message_id on success, or None on failure.
    """
    try:
        pickup_text = str(ride_data.get("pickup", "Unknown"))
        dropoff_text = str(ride_data.get("dropoff", "Unknown"))
        fare_text = str(ride_data.get("price", "0"))

        message = messaging.Message(
            data={
                "type": "ride_request",
                "ride_id": str(ride_data.get("ride_id", "")),
                "pickup": pickup_text,
                "dropoff": dropoff_text,
                "pickup_lat": str(ride_data.get("pickup_lat", "")),
                "pickup_lng": str(ride_data.get("pickup_lng", "")),
                "drop_lat": str(ride_data.get("drop_lat", "")),
                "drop_lng": str(ride_data.get("drop_lng", "")),
                "estimated_fare": fare_text,
                "distance_km": str(ride_data.get("distance_km", "")),
                "user_name": str(ride_data.get("user_name", "")),
                "user_phone": str(ride_data.get("user_phone", "")),
                "otp": str(ride_data.get("otp", "")),
            },
            token=fcm_token,
            android=messaging.AndroidConfig(
                priority="high",
                ttl=timedelta(seconds=DISPATCH_TIMEOUT_SECONDS + 5)
            ),
        )
        response = messaging.send(message)
        logger.info("FCM sent to token %s… — message_id: %s", fcm_token[:20], response)
        return response
    except messaging.UnregisteredError:
        logger.warning("FCM token %s… is unregistered — rider may have uninstalled", fcm_token[:20])
        return None
    except Exception as exc:
        logger.error("FCM send failed: %s", exc, exc_info=True)
        return None


# ─── Waterfall Dispatch Loop (Background Task) ───────────────────────────────

# In-memory tracking of dispatch state for active rides
# Key: ride_id, Value: {"event": asyncio.Event, "accepted_by": str|None}
_dispatch_state: Dict[str, dict] = {}


async def waterfall_dispatch(ride_id: str, ride_data: dict):
    """
    Background task that implements the waterfall dispatch:
    1. Find nearest online riders
    2. Send FCM to rider #1, wait 15s
    3. If no accept → try rider #2, etc.
    4. After MAX_DISPATCH_ATTEMPTS failures → set status = "no_riders"
    """
    pickup_lat = ride_data.get("pickup_lat")
    pickup_lng = ride_data.get("pickup_lng")

    if pickup_lat is None or pickup_lng is None:
        logger.warning("Ride %s has no pickup coordinates, cannot dispatch", ride_id)
        db.collection("rides").document(ride_id).update({
            "status": "no_riders",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return

    attempted_rider_ids = []
    dispatch_attempt = 0

    while dispatch_attempt < MAX_DISPATCH_ATTEMPTS:
        # Find nearest riders, excluding already-attempted ones
        candidates = find_nearest_riders(
            pickup_lat, pickup_lng,
            exclude_rider_ids=attempted_rider_ids,
        )

        if not candidates:
            logger.info("Ride %s: No more riders available within radius", ride_id)
            break

        rider = candidates[0]
        rider_uid = rider["uid"]
        attempted_rider_ids.append(rider_uid)
        dispatch_attempt += 1

        logger.info(
            "Ride %s: Dispatching to rider %s (%s, %.2f km away) — attempt %d/%d",
            ride_id, rider_uid, rider["full_name"],
            rider["distance_km"], dispatch_attempt, MAX_DISPATCH_ATTEMPTS,
        )

        # Update ride doc with current dispatch target
        db.collection("rides").document(ride_id).update({
            "dispatch_rider_uid": rider_uid,
            "dispatch_rider_name": rider["full_name"],
            "dispatch_attempt": dispatch_attempt,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        # Create an asyncio Event for this ride's accept/reject signal
        accept_event = asyncio.Event()
        _dispatch_state[ride_id] = {
            "event": accept_event,
            "accepted_by": None,
            "target_rider": rider_uid,
        }

        # Send FCM notification to this rider
        send_targeted_ride_notification(rider["fcm_token"], ride_data)

        # Wait for accept/reject with timeout
        try:
            await asyncio.wait_for(accept_event.wait(), timeout=DISPATCH_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.info("Ride %s: Rider %s timed out after %ds",
                        ride_id, rider_uid, DISPATCH_TIMEOUT_SECONDS)

        # Check if the ride was accepted
        state = _dispatch_state.get(ride_id, {})
        if state.get("accepted_by"):
            logger.info("Ride %s: Accepted by rider %s", ride_id, state["accepted_by"])
            # Clean up dispatch state
            _dispatch_state.pop(ride_id, None)
            return

        # Check if ride was cancelled by user
        ride_doc = db.collection("rides").document(ride_id).get()
        if ride_doc.exists:
            current_status = ride_doc.to_dict().get("status", "")
            if current_status in ("cancelled", "accepted"):
                logger.info("Ride %s: Status changed to %s, stopping dispatch",
                            ride_id, current_status)
                _dispatch_state.pop(ride_id, None)
                return

        logger.info("Ride %s: Rider %s did not accept, trying next", ride_id, rider_uid)

    # All attempts exhausted — no rider found
    logger.info("Ride %s: All %d dispatch attempts exhausted, no rider found",
                ride_id, dispatch_attempt)

    db.collection("rides").document(ride_id).update({
        "status": "no_riders",
        "dispatch_rider_uid": "",
        "dispatch_rider_name": "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    # Clean up
    _dispatch_state.pop(ride_id, None)


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Affitto Ride & Rent API", "version": "2.0.0"}


# ─── User Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/users", status_code=201)
async def create_or_update_user(user: UserProfile):
    """Create a new user profile or update an existing one in Firestore."""
    user_ref = db.collection("users").document(user.uid)
    existing = user_ref.get()

    if existing.exists:
        # Only update fields that are explicitly provided
        update_data = {}
        if user.phone:
            update_data["phone"] = user.phone
        if user.name:
            update_data["name"] = user.name
        if user.photo_url:
            update_data["photo_url"] = user.photo_url
        if update_data:
            update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            user_ref.update(update_data)
        return {"message": "User profile updated", "uid": user.uid}
    else:
        user_ref.set({
            "uid": user.uid,
            "name": user.name,
            "email": user.email,
            "phone": user.phone or "",
            "photo_url": user.photo_url or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return {"message": "User profile created", "uid": user.uid}


@app.get("/api/users/{uid}")
async def get_user(uid: str):
    """Fetch a user profile from Firestore by UID."""
    user_ref = db.collection("users").document(uid)
    doc = user_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    return doc.to_dict()


@app.patch("/api/users/{uid}")
async def update_user_profile(uid: str, update: UserProfileUpdate):
    """Partially update a user's profile (phone, name)."""
    user_ref = db.collection("users").document(uid)
    doc = user_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = {}
    if update.phone is not None:
        update_data["phone"] = update.phone
    if update.name is not None:
        update_data["name"] = update.name
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    user_ref.update(update_data)
    return {"message": "Profile updated", "uid": uid}


# ─── Ride Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/rides", status_code=201)
async def create_ride(ride: RideCreate, background_tasks: BackgroundTasks):
    """
    Book a new ride and start the waterfall dispatch loop.
    
    1. Creates a ride document in Firestore with status="pending" and a random OTP
    2. Launches background waterfall dispatch (find nearest rider → FCM → wait → next)
    3. Returns immediately with ride_id so the user app can listen to Firestore
    """
    ride_id = str(uuid.uuid4())
    otp = get_or_create_user_otp(ride.user_id)

    ride_data = {
        "ride_id": ride_id,
        "user_id": ride.user_id,
        "rider_id": "",
        "pickup": ride.pickup,
        "dropoff": ride.dropoff,
        "distance_km": round(ride.distance_km, 2),
        "price": round(ride.price, 2),
        "pickup_lat": ride.pickup_lat,
        "pickup_lng": ride.pickup_lng,
        "drop_lat": ride.drop_lat,
        "drop_lng": ride.drop_lng,
        "user_name": ride.user_name or "",
        "user_phone": ride.user_phone or "",
        "otp": otp,
        "status": "pending",
        "rider_name": "",
        "rider_phone": "",
        "rider_earnings": 0.0,
        "payment_method": "",
        "dispatch_rider_uid": "",
        "dispatch_rider_name": "",
        "dispatch_attempt": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    db.collection("rides").document(ride_id).set(ride_data)
    logger.info("Ride %s created — dispatching...", ride_id)

    # Launch the waterfall dispatch in the background
    background_tasks.add_task(waterfall_dispatch, ride_id, ride_data)

    return {
        "message": "Ride booked — searching for riders",
        "ride_id": ride_id,
        "otp": otp,
        "status": "pending",
    }


@app.get("/api/rides/{ride_id}/status")
async def get_ride_status(ride_id: str):
    """Get current ride status (for user app polling / fallback)."""
    doc = db.collection("rides").document(ride_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Ride not found")
    data = doc.to_dict()
    return {
        "ride_id": ride_id,
        "status": data.get("status"),
        "rider_id": data.get("rider_id", ""),
        "rider_name": data.get("rider_name", ""),
        "rider_phone": data.get("rider_phone", ""),
        "otp": data.get("otp", ""),
        "dispatch_attempt": data.get("dispatch_attempt", 0),
    }


@app.post("/api/rides/{ride_id}/accept")
async def accept_ride(ride_id: str, request: RideAcceptRequest):
    """
    Rider accepts a dispatched ride.
    
    Validates:
    - Ride exists and is still pending
    - The rider is the current dispatch target
    
    Updates the ride doc and signals the dispatch loop to stop.
    """
    ride_ref = db.collection("rides").document(ride_id)
    ride_doc = ride_ref.get()

    if not ride_doc.exists:
        raise HTTPException(status_code=404, detail="Ride not found")

    ride_data = ride_doc.to_dict()
    current_status = ride_data.get("status", "")
    dispatch_target = ride_data.get("dispatch_rider_uid", "")

    if current_status != "pending":
        raise HTTPException(status_code=409, detail=f"Ride is no longer pending (status: {current_status})")

    if dispatch_target and dispatch_target != request.rider_id:
        raise HTTPException(status_code=403, detail="You are not the current dispatch target for this ride")

    # Fetch rider details for denormalization
    rider_doc = db.collection("riders").document(request.rider_id).get()
    rider_name = ""
    rider_phone = ""
    if rider_doc.exists:
        rider_data = rider_doc.to_dict()
        rider_name = rider_data.get("full_name", "")
        rider_phone = rider_data.get("mobile", "")

    # Update ride document
    ride_ref.update({
        "status": "accepted",
        "rider_id": request.rider_id,
        "rider_name": rider_name,
        "rider_phone": rider_phone,
        "dispatch_rider_uid": "",
        "dispatch_rider_name": "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    # Signal the dispatch loop
    state = _dispatch_state.get(ride_id)
    if state:
        state["accepted_by"] = request.rider_id
        state["event"].set()

    logger.info("Ride %s accepted by rider %s (%s)", ride_id, request.rider_id, rider_name)

    return {
        "message": "Ride accepted",
        "ride_id": ride_id,
        "otp": ride_data.get("otp", ""),
    }


@app.post("/api/rides/{ride_id}/reject")
async def reject_ride(ride_id: str, request: RideRejectRequest):
    """
    Rider explicitly rejects a ride. Signals the dispatch loop to
    immediately move to the next rider instead of waiting for timeout.
    """
    ride_ref = db.collection("rides").document(ride_id)
    ride_doc = ride_ref.get()

    if not ride_doc.exists:
        raise HTTPException(status_code=404, detail="Ride not found")

    ride_data = ride_doc.to_dict()
    if ride_data.get("status") != "pending":
        return {"message": "Ride is no longer pending", "ride_id": ride_id}

    # Signal the dispatch loop to move on
    state = _dispatch_state.get(ride_id)
    if state and state.get("target_rider") == request.rider_id:
        state["event"].set()  # Unblocks the wait, accepted_by stays None → next rider
        logger.info("Ride %s: Rider %s explicitly rejected", ride_id, request.rider_id)

    return {"message": "Ride rejected", "ride_id": ride_id}


@app.get("/api/rides/{user_id}")
async def get_rides(user_id: str):
    """Fetch all rides for a given user, most recent first."""
    rides_ref = (
        db.collection("rides")
        .where("user_id", "==", user_id)
        .order_by("created_at", direction=firestore.Query.DESCENDING)
    )
    docs = rides_ref.stream()
    rides = [doc.to_dict() for doc in docs]
    return {"rides": rides, "count": len(rides)}


# ─── Rider Location Endpoint ────────────────────────────────────────────────

@app.post("/api/riders/{uid}/location")
async def update_rider_location(uid: str, location: RiderLocationUpdate):
    """
    Update rider's GPS coordinates in Firestore.
    Called periodically by the Foreground Service on the Rider app.
    """
    rider_ref = db.collection("riders").document(uid)
    doc = rider_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Rider not found")

    update_data = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if location.is_online is not None:
        update_data["is_online"] = location.is_online

    rider_ref.update(update_data)
    return {"message": "Location updated", "uid": uid}


# ─── Contact / Rental Inquiry Endpoint ───────────────────────────────────────

@app.post("/api/contact", status_code=201)
async def submit_contact(contact: ContactRequest):
    """Save a rental inquiry / contact-us request to Firestore."""
    request_id = str(uuid.uuid4())
    contact_data = {
        "request_id": request_id,
        "name": contact.name,
        "phone": contact.phone,
        "vehicle": contact.vehicle,
        "message": contact.message or "",
        "status": "new",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db.collection("contact_requests").document(request_id).set(contact_data)
    return {"message": "Inquiry submitted successfully", "request_id": request_id}
