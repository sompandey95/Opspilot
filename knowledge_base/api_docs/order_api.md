# ShopEasy Mock Order Service API

**Base URL:** `http://localhost:8001`
**Version:** v1
**Auth:** None (internal service — not exposed to the internet)
**Content-Type:** `application/json`

---

## Overview

The Order Service is an internal mock microservice that the OpsPilot agent calls to read order state, check refund eligibility, process refunds, cancel orders, and retrieve customer profiles. All amounts are in **Indian Rupees (INR)**. Dates are ISO 8601 in IST (UTC+5:30).

---

## Endpoints

### 1. GET /orders/{order_id}

Retrieve complete details of a single order.

**Path parameters:**
| Parameter | Type | Description |
|---|---|---|
| `order_id` | string | ShopEasy order ID (format: SE-XXXXXXX) |

**Example request:**
```
GET /orders/SE-9834112
```

**Example response (200 OK):**
```json
{
  "order_id": "SE-9834112",
  "customer_id": "CUST-441829",
  "status": "delivered",
  "items": [
    {
      "sku": "SONY-WH1000XM4-BLK",
      "name": "Sony WH-1000XM4 Wireless Headphones (Black)",
      "quantity": 1,
      "unit_price": 4599.00,
      "seller_id": "SELL-10923",
      "seller_name": "AudioWorld India"
    }
  ],
  "payment": {
    "method": "upi",
    "upi_vpa": "priya.sharma@okicici",
    "amount_paid": 4599.00,
    "cod_fee": 0.00,
    "express_fee": 0.00,
    "gst_amount": 828.0,
    "transaction_id": "ICIC26061900345678"
  },
  "shipping_address": {
    "name": "Priya Sharma",
    "line1": "Flat 402, Sunrise Apartments",
    "line2": "Sector 18",
    "city": "Noida",
    "state": "Uttar Pradesh",
    "pincode": "201301",
    "phone": "9876543210"
  },
  "delivery": {
    "courier": "Delhivery",
    "awb": "DEL7782901334",
    "dispatched_at": "2026-06-17T10:30:00+05:30",
    "delivered_at": "2026-06-19T15:42:13+05:30",
    "delivery_otp_required": true,
    "delivery_otp_confirmed": false
  },
  "created_at": "2026-06-17T09:15:44+05:30",
  "updated_at": "2026-06-19T15:42:15+05:30"
}
```

**Error responses:**
- `404 Not Found` — Order ID not found.
- `400 Bad Request` — Malformed order ID format.

---

### 2. GET /orders/{order_id}/tracking

Get live shipment tracking events for an order.

**Path parameters:**
| Parameter | Type | Description |
|---|---|---|
| `order_id` | string | ShopEasy order ID |

**Example request:**
```
GET /orders/SE-9834112/tracking
```

**Example response (200 OK):**
```json
{
  "order_id": "SE-9834112",
  "awb": "DEL7782901334",
  "courier": "Delhivery",
  "current_status": "delivered",
  "estimated_delivery": "2026-06-19",
  "events": [
    {
      "timestamp": "2026-06-17T11:05:00+05:30",
      "status": "dispatched",
      "location": "ShopEasy Fulfilment Centre, Bhiwandi, Maharashtra",
      "description": "Shipment picked up by Delhivery"
    },
    {
      "timestamp": "2026-06-17T22:30:00+05:30",
      "status": "in_transit",
      "location": "Delhivery Hub, Bhiwandi, Maharashtra",
      "description": "In transit to destination city"
    },
    {
      "timestamp": "2026-06-18T06:15:00+05:30",
      "status": "in_transit",
      "location": "Delhivery Hub, Noida, Uttar Pradesh",
      "description": "Arrived at destination hub"
    },
    {
      "timestamp": "2026-06-19T09:00:00+05:30",
      "status": "out_for_delivery",
      "location": "Delhivery Delivery Centre, Sector 63, Noida",
      "description": "Out for delivery"
    },
    {
      "timestamp": "2026-06-19T15:42:13+05:30",
      "status": "delivered",
      "location": "Sector 18, Noida - 201301",
      "description": "Delivered (OTP not confirmed)"
    }
  ]
}
```

---

### 3. POST /orders/{order_id}/refund/check

Check whether a refund can be initiated for an order and what amount is eligible.

**Path parameters:**
| Parameter | Type | Description |
|---|---|---|
| `order_id` | string | ShopEasy order ID |

**Request body:**
```json
{
  "reason": "wrong_item"
}
```

**Reason codes:**
| Code | Description |
|---|---|
| `defective` | Item received is defective or damaged |
| `wrong_item` | Received a different item than ordered |
| `not_as_described` | Item materially differs from listing |
| `change_of_mind` | Customer no longer wants the item |
| `not_delivered` | Item marked delivered but not received |
| `seller_cancelled` | Seller cancelled the order |

**Example request:**
```
POST /orders/SE-9012445/refund/check
Content-Type: application/json

{
  "reason": "wrong_item"
}
```

**Example response (200 OK):**
```json
{
  "order_id": "SE-9012445",
  "eligible": true,
  "refund_amount": 3199.00,
  "restocking_fee": 0.00,
  "courtesy_coupon": 50.00,
  "reason": "wrong_item",
  "within_return_window": true,
  "return_window_days": 10,
  "days_since_delivery": 3,
  "refund_to": "axis_credit_card",
  "estimated_refund_days": "5-7 business days",
  "notes": "Full refund applicable. Wrong item return pickup will be arranged. ₹50 courtesy coupon will be added."
}
```

**Ineligible response (200 OK, eligible=false):**
```json
{
  "order_id": "SE-8001234",
  "eligible": false,
  "reason": "change_of_mind",
  "within_return_window": false,
  "days_since_delivery": 15,
  "return_window_days": 10,
  "notes": "Return window of 10 days has expired. Item delivered on 2026-06-05, today is 2026-06-20."
}
```

---

### 4. POST /orders/{order_id}/refund/process

Process a refund for an order. Should only be called after `/refund/check` confirms eligibility.

**Path parameters:**
| Parameter | Type | Description |
|---|---|---|
| `order_id` | string | ShopEasy order ID |

**Request body:**
```json
{
  "amount": 3199.00,
  "reason": "wrong_item",
  "refund_to": "original_payment_method",
  "agent_id": "AGENT-0042",
  "notes": "Wrong item received. Return pickup scheduled."
}
```

**Fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `amount` | float | Yes | Refund amount in INR |
| `reason` | string | Yes | Reason code (see /refund/check) |
| `refund_to` | string | Yes | `original_payment_method`, `shopeasy_wallet`, `bank_neft` |
| `bank_account` | string | No | Required if `refund_to` = `bank_neft` |
| `ifsc` | string | No | Required if `refund_to` = `bank_neft` |
| `agent_id` | string | Yes | ID of the support agent initiating the refund |
| `notes` | string | No | Internal notes for audit trail |

**Example request:**
```
POST /orders/SE-9012445/refund/process
Content-Type: application/json

{
  "amount": 3199.00,
  "reason": "wrong_item",
  "refund_to": "original_payment_method",
  "agent_id": "AGENT-0042",
  "notes": "Wrong item confirmed via photos. Return pickup June 10."
}
```

**Example response (200 OK):**
```json
{
  "refund_id": "REF-20260609-009012445",
  "order_id": "SE-9012445",
  "status": "initiated",
  "amount": 3199.00,
  "refund_to": "axis_credit_card_4412",
  "estimated_credit": "2026-06-16",
  "courtesy_coupon_code": "WRONG50JUN26",
  "courtesy_coupon_value": 50.00,
  "initiated_at": "2026-06-09T10:05:33+05:30",
  "initiated_by": "AGENT-0042"
}
```

**Error responses:**
- `409 Conflict` — Refund already processed for this order.
- `422 Unprocessable Entity` — Amount exceeds order value or reason not valid.
- `400 Bad Request` — Missing required fields.

---

### 5. POST /orders/{order_id}/cancel

Cancel an order. Validates that cancellation is possible given current order status.

**Path parameters:**
| Parameter | Type | Description |
|---|---|---|
| `order_id` | string | ShopEasy order ID |

**Request body:**
```json
{
  "reason": "customer_request",
  "agent_id": "AGENT-0042",
  "notes": "Customer found better price elsewhere."
}
```

**Reason codes:** `customer_request`, `seller_cancelled`, `system_error`, `duplicate_order`, `payment_failure`

**Example request:**
```
POST /orders/SE-9301847/cancel
Content-Type: application/json

{
  "reason": "customer_request",
  "agent_id": "AGENT-0042",
  "notes": "Customer changed mind. Order not yet dispatched."
}
```

**Example response (200 OK):**
```json
{
  "order_id": "SE-9301847",
  "status": "cancelled",
  "cancellation_id": "CNCL-20260614-9301847",
  "refund": {
    "amount": 2150.00,
    "refund_to": "kotak_credit_card_7823",
    "estimated_credit": "2026-06-21",
    "status": "initiated"
  },
  "cancelled_at": "2026-06-14T08:50:12+05:30"
}
```

**Error response (409 Conflict) — already dispatched:**
```json
{
  "error": "cancellation_not_possible",
  "message": "Order SE-9712445 is in 'dispatched' status and cannot be cancelled. Advise customer to refuse delivery.",
  "current_status": "dispatched",
  "dispatched_at": "2026-06-19T06:45:00+05:30"
}
```

---

### 6. GET /customers/{customer_id}

Retrieve customer profile and account standing information.

**Path parameters:**
| Parameter | Type | Description |
|---|---|---|
| `customer_id` | string | ShopEasy customer ID (format: CUST-XXXXXX) |

**Example request:**
```
GET /customers/CUST-441829
```

**Example response (200 OK):**
```json
{
  "customer_id": "CUST-441829",
  "name": "Priya Sharma",
  "email": "priya.sharma@gmail.com",
  "phone": "9876543210",
  "registered_at": "2024-03-15T10:30:00+05:30",
  "account_status": "active",
  "kyc_verified": true,
  "addresses": [
    {
      "address_id": "ADDR-001",
      "label": "Home",
      "line1": "Flat 402, Sunrise Apartments",
      "line2": "Sector 18",
      "city": "Noida",
      "state": "Uttar Pradesh",
      "pincode": "201301",
      "is_default": true
    }
  ],
  "wallet_balance": 150.00,
  "loyalty_tier": "Silver",
  "stats": {
    "total_orders": 34,
    "total_spend_inr": 47820.00,
    "cancellations_this_month": 2,
    "cancellation_limit": 5,
    "cod_eligible": true,
    "cod_return_rate_90d": 0.05,
    "open_refunds": 1,
    "open_disputes": 0
  },
  "flags": []
}
```

**Flagged account example (partial):**
```json
{
  "customer_id": "CUST-778234",
  "account_status": "under_review",
  "flags": ["cancellation_limit_exceeded"],
  "stats": {
    "cancellations_this_month": 6,
    "cancellation_limit": 5,
    "cod_eligible": false
  }
}
```

**Error responses:**
- `404 Not Found` — Customer ID not found.

---

## Notes for OpsPilot Agent

- Always call `/refund/check` before `/refund/process` — never skip eligibility validation.
- When calling `/orders/{order_id}/cancel`, check the response for `cancellation_not_possible` (409) and advise the customer to refuse delivery instead.
- For COD refunds to bank, always collect and pass `bank_account` + `ifsc` in the process call.
- Customer `stats.cancellations_this_month` vs `stats.cancellation_limit` tells you whether a cancellation will trigger an account review before you attempt it.
- `delivery_otp_confirmed: false` on a "delivered" order is a strong signal for a missing delivery investigation — escalate.
