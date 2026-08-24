"""Deterministic seed data: ~150 customers, 500 orders with realistic Indian data.

Everything derives from random.seed(42) and a fixed ANCHOR date, so the same
IDs, amounts, and statuses appear on every startup. Eval scenarios reference
the pinned orders below, so their IDs and semantics form part of the dataset.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from mock_services.order_service.models import (
    Customer,
    Order,
    OrderStatus,
    PaymentMethod,
    Refund,
    RefundStatus,
)

# Fixed "today" for deterministic relative dates (delivered 2 days ago, etc.)
ANCHOR = date(2026, 8, 1)

_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan",
    "Rohan", "Kabir", "Ananya", "Diya", "Aadhya", "Saanvi", "Priya", "Isha",
    "Meera", "Kavya", "Sneha", "Pooja", "Rahul", "Amit", "Vikram", "Suresh",
    "Deepak", "Neha", "Ritu", "Swati", "Anjali", "Lakshmi",
]
_LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Mehta", "Patel", "Reddy", "Nair", "Iyer",
    "Singh", "Kumar", "Das", "Chatterjee", "Banerjee", "Joshi", "Desai",
    "Kulkarni", "Rao", "Menon", "Agarwal", "Malhotra",
]
_CITIES = [
    ("Mumbai", "400001"), ("Delhi", "110001"), ("Bengaluru", "560001"),
    ("Hyderabad", "500001"), ("Chennai", "600001"), ("Kolkata", "700001"),
    ("Pune", "411001"), ("Ahmedabad", "380001"), ("Jaipur", "302001"),
    ("Lucknow", "226001"), ("Kochi", "682001"), ("Indore", "452001"),
    ("Nagpur", "440001"), ("Patna", "800001"), ("Bhopal", "462001"),
]
# (product, category, min ₹, max ₹)
_PRODUCTS = [
    ("boAt Airdopes 141 TWS Earbuds", "electronics", 999, 1499),
    ("Samsung Galaxy M35 5G", "electronics", 14999, 18999),
    ("Noise ColorFit Pro 4 Smartwatch", "electronics", 1799, 2999),
    ("HP 15s Laptop (Ryzen 5)", "electronics", 39999, 45999),
    ("Mi 20000mAh Power Bank", "electronics", 1499, 2199),
    ("Allen Solly Men's Polo T-shirt", "clothing", 599, 1099),
    ("Biba Women's Cotton Kurta", "clothing", 899, 1899),
    ("Puma Running Shoes", "clothing", 1999, 3499),
    ("Fabindia Bedsheet Set (King)", "home", 1299, 2499),
    ("Prestige Induction Cooktop", "home", 1899, 2899),
    ("Milton Thermosteel Flask 1L", "home", 699, 999),
    ("Atomic Habits (Paperback)", "books", 349, 499),
    ("Classmate Notebook Pack of 6", "books", 249, 349),
    ("LEGO Classic Bricks Box", "toys", 1499, 2499),
    ("Yonex Badminton Racket", "sports", 999, 1999),
    ("Lakme 9to5 Lipstick", "beauty", 399, 599),
    ("Mamaearth Face Wash Combo", "beauty", 449, 699),
]

_STATUS_WEIGHTS = [
    (OrderStatus.DELIVERED, 60),
    (OrderStatus.IN_TRANSIT, 18),
    (OrderStatus.DELAYED, 10),
    (OrderStatus.PROCESSING, 4),
    (OrderStatus.CANCELLED, 4),
    (OrderStatus.REFUNDED, 4),
]
_PAYMENT_WEIGHTS = [
    (PaymentMethod.UPI, 45),
    (PaymentMethod.CARD, 20),
    (PaymentMethod.COD, 20),
    (PaymentMethod.NETBANKING, 10),
    (PaymentMethod.WALLET, 5),
]


def _weighted(rng: random.Random, pairs):
    values, weights = zip(*pairs)
    return rng.choices(values, weights=weights, k=1)[0]


def _make_customers(rng: random.Random, count: int = 150) -> list[Customer]:
    customers: list[Customer] = []
    seen_emails: set[str] = set()
    i = 0
    while len(customers) < count:
        i += 1
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        if email in seen_emails:
            continue
        seen_emails.add(email)
        city, pincode = rng.choice(_CITIES)
        customers.append(
            Customer(
                customer_id=f"CUST-{10000 + i}",
                name=f"{first} {last}",
                email=email,
                phone=f"+91-9{rng.randint(100000000, 999999999)}",
                city=city,
                pincode=pincode,
                plus_member=rng.random() < 0.25,
            )
        )
    return customers


def _make_order(rng: random.Random, order_num: int, customer: Customer) -> Order:
    product, category, lo, hi = rng.choice(_PRODUCTS)
    amount = float(rng.randrange(lo, hi + 1))
    status = _weighted(rng, _STATUS_WEIGHTS)
    payment = _weighted(rng, _PAYMENT_WEIGHTS)
    ordered_on = ANCHOR - timedelta(days=rng.randint(1, 90))

    delivered_on = None
    delivery_eta = None
    refund = None
    order_id = f"ORD-2024-{order_num}"

    if status == OrderStatus.DELIVERED:
        delivered_on = ordered_on + timedelta(days=rng.randint(2, 7))
        if delivered_on > ANCHOR:
            delivered_on = ANCHOR
    elif status in (OrderStatus.IN_TRANSIT, OrderStatus.PROCESSING):
        delivery_eta = ANCHOR + timedelta(days=rng.randint(1, 6))
    elif status == OrderStatus.DELAYED:
        delivery_eta = ANCHOR + timedelta(days=rng.randint(2, 8))
    elif status == OrderStatus.REFUNDED:
        delivered_on = ordered_on + timedelta(days=rng.randint(2, 7))
        refund = Refund(
            refund_id=f"REF-{order_num}",
            order_id=order_id,
            amount_inr=amount,
            reason=rng.choice(["defective", "wrong_item", "not_as_described"]),
            status=RefundStatus.COMPLETED,
            initiated_on=delivered_on + timedelta(days=rng.randint(1, 5)),
        )

    return Order(
        order_id=order_id,
        customer_id=customer.customer_id,
        product_name=product,
        category=category,
        amount_inr=amount,
        payment_method=payment,
        status=status,
        ordered_on=ordered_on,
        delivered_on=delivered_on,
        delivery_eta=delivery_eta,
        shipping_city=customer.city,
        shipping_pincode=customer.pincode,
        refund=refund,
    )


def _pinned_orders(customers: list[Customer]) -> list[Order]:
    """Hand-pinned orders that eval scenarios and docs reference by ID."""
    c1, c2, c3, c4, c5, c6 = customers[:6]
    return [
        # Delayed, UPI — the canonical "refund chahiye" multi-step scenario
        Order(
            order_id="ORD-2024-55001", customer_id=c1.customer_id,
            product_name="boAt Airdopes 141 TWS Earbuds", category="electronics",
            amount_inr=1499.00, payment_method=PaymentMethod.UPI,
            status=OrderStatus.DELAYED,
            ordered_on=ANCHOR - timedelta(days=12),
            delivery_eta=ANCHOR + timedelta(days=4),
            shipping_city=c1.city, shipping_pincode=c1.pincode,
        ),
        # Delayed card order used by the approval-flow walkthrough.
        Order(
            order_id="ORD-2024-78432", customer_id=c2.customer_id,
            product_name="Noise ColorFit Pro 4 Smartwatch", category="electronics",
            amount_inr=1299.00, payment_method=PaymentMethod.CARD,
            status=OrderStatus.DELAYED,
            ordered_on=ANCHOR - timedelta(days=10),
            delivery_eta=ANCHOR + timedelta(days=3),
            shipping_city=c2.city, shipping_pincode=c2.pincode,
        ),
        # Delivered 2 days ago — inside the return window
        Order(
            order_id="ORD-2024-51234", customer_id=c3.customer_id,
            product_name="Biba Women's Cotton Kurta", category="clothing",
            amount_inr=1199.00, payment_method=PaymentMethod.UPI,
            status=OrderStatus.DELIVERED,
            ordered_on=ANCHOR - timedelta(days=6),
            delivered_on=ANCHOR - timedelta(days=2),
            shipping_city=c3.city, shipping_pincode=c3.pincode,
        ),
        # Delivered 45 days ago — return window closed
        Order(
            order_id="ORD-2024-52000", customer_id=c4.customer_id,
            product_name="Prestige Induction Cooktop", category="home",
            amount_inr=2299.00, payment_method=PaymentMethod.NETBANKING,
            status=OrderStatus.DELIVERED,
            ordered_on=ANCHOR - timedelta(days=50),
            delivered_on=ANCHOR - timedelta(days=45),
            shipping_city=c4.city, shipping_pincode=c4.pincode,
        ),
        # Cancelled with completed refund
        Order(
            order_id="ORD-2024-53000", customer_id=c5.customer_id,
            product_name="Samsung Galaxy M35 5G", category="electronics",
            amount_inr=16999.00, payment_method=PaymentMethod.CARD,
            status=OrderStatus.CANCELLED,
            ordered_on=ANCHOR - timedelta(days=20),
            shipping_city=c5.city, shipping_pincode=c5.pincode,
            refund=Refund(
                refund_id="REF-53000", order_id="ORD-2024-53000",
                amount_inr=16999.00, reason="seller_cancellation",
                status=RefundStatus.COMPLETED,
                initiated_on=ANCHOR - timedelta(days=18),
            ),
        ),
        # In transit, COD, on time — NOT refund-eligible
        Order(
            order_id="ORD-2024-54000", customer_id=c6.customer_id,
            product_name="Milton Thermosteel Flask 1L", category="home",
            amount_inr=849.00, payment_method=PaymentMethod.COD,
            status=OrderStatus.IN_TRANSIT,
            ordered_on=ANCHOR - timedelta(days=3),
            delivery_eta=ANCHOR + timedelta(days=2),
            shipping_city=c6.city, shipping_pincode=c6.pincode,
        ),
    ]


def build_store(n_orders: int = 500, seed: int = 42) -> tuple[dict[str, Customer], dict[str, Order]]:
    """Build the full deterministic store: {customer_id: Customer}, {order_id: Order}."""
    rng = random.Random(seed)
    customers = _make_customers(rng)

    orders: dict[str, Order] = {}
    for order in _pinned_orders(customers):
        orders[order.order_id] = order

    order_num = 60000
    while len(orders) < n_orders:
        order_num += rng.randint(1, 17)
        customer = rng.choice(customers)
        order = _make_order(rng, order_num, customer)
        orders[order.order_id] = order

    return {c.customer_id: c for c in customers}, orders
