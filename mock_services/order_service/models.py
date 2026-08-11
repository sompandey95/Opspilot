"""Pydantic models for the mock ShopEasy order/customer service.

Deliberately in-memory (no database): the store is rebuilt deterministically
from seed.py on every startup, so eval scenarios can rely on stable order IDs
and a restart cleanly resets any refunds/cancellations made during a run.
"""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel


class OrderStatus(str, Enum):
    PROCESSING = "processing"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentMethod(str, Enum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    COD = "cod"
    WALLET = "wallet"


class RefundStatus(str, Enum):
    NONE = "none"
    PROCESSING = "processing"
    COMPLETED = "completed"


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str
    phone: str
    city: str
    pincode: str
    plus_member: bool = False


class Refund(BaseModel):
    refund_id: str
    order_id: str
    amount_inr: float
    reason: str
    status: RefundStatus
    initiated_on: date


class Order(BaseModel):
    order_id: str
    customer_id: str
    product_name: str
    category: str
    amount_inr: float
    payment_method: PaymentMethod
    status: OrderStatus
    ordered_on: date
    delivered_on: date | None = None
    delivery_eta: date | None = None
    shipping_city: str
    shipping_pincode: str
    refund: Refund | None = None


class RefundEligibility(BaseModel):
    order_id: str
    eligible: bool
    amount_inr: float
    reason: str


class RefundRequest(BaseModel):
    amount_inr: float | None = None  # default: full eligible amount
    reason: str


class CancelRequest(BaseModel):
    reason: str
