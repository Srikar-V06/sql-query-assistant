"""
Database layer for the SQL Query Assistant environment.

Creates an in-memory SQLite database seeded with realistic e-commerce data:
  - customers       (500 rows)
  - products        (50 rows across 5 categories)
  - orders          (1200 rows spanning 2022–2024)
  - order_items     (3000+ rows)
  - reviews         (800 rows)

The same SEED value always produces identical data, making every
reset() call deterministic and baseline scores reproducible.
"""

import sqlite3
import random
from datetime import date, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Seed — never change this. Reproducibility depends on it.
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Static lookup data (no external deps needed)
# ---------------------------------------------------------------------------
CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal",
]

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
    "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
    "Priya", "Ananya", "Isha", "Shreya", "Pooja",
    "Divya", "Neha", "Riya", "Kavya", "Meera",
    "Rohit", "Amit", "Suresh", "Rahul", "Vikram",
    "Sunita", "Geeta", "Rekha", "Sonal", "Nisha",
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Singh", "Kumar",
    "Gupta", "Joshi", "Mehta", "Shah", "Nair",
    "Reddy", "Iyer", "Pillai", "Chopra", "Bose",
    "Das", "Mishra", "Tiwari", "Yadav", "Pandey",
]

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Sports"]

PRODUCT_NAMES = {
    "Electronics": [
        "Wireless Earbuds Pro", "Smart Watch Series X", "USB-C Hub 7-in-1",
        "Portable Charger 20000mAh", "Bluetooth Speaker Mini",
        "Laptop Stand Aluminium", "Mechanical Keyboard TKL",
        "Webcam 1080p HD", "LED Desk Lamp Smart", "SSD 1TB External",
    ],
    "Clothing": [
        "Cotton Kurta Classic", "Denim Jeans Slim Fit", "Formal Shirt White",
        "Sports T-Shirt Dri-Fit", "Woolen Sweater Round Neck",
        "Casual Sneakers", "Leather Belt Brown", "Canvas Tote Bag",
        "Chino Trousers Beige", "Ethnic Saree Silk",
    ],
    "Books": [
        "Clean Code", "The Pragmatic Programmer", "Atomic Habits",
        "Deep Work", "System Design Interview Vol 2",
        "Python Crash Course", "The Lean Startup", "Zero to One",
        "Thinking Fast and Slow", "The Alchemist",
    ],
    "Home & Kitchen": [
        "Stainless Steel Pressure Cooker 5L", "Non-Stick Tawa 30cm",
        "Air Fryer 4L Digital", "Water Purifier RO UV",
        "Mixer Grinder 750W", "Induction Cooktop 2000W",
        "Chopping Board Bamboo Set", "Glass Storage Jar Set 6pc",
        "Dish Drying Rack Foldable", "Electric Kettle 1.8L",
    ],
    "Sports": [
        "Yoga Mat 6mm Anti-Slip", "Resistance Bands Set 5pc",
        "Dumbbell Set Adjustable 10kg", "Jump Rope Speed",
        "Badminton Racket Pro", "Cricket Bat Kashmir Willow",
        "Cycling Gloves Padded", "Gym Bag Large Capacity",
        "Foam Roller Deep Tissue", "Water Bottle Insulated 1L",
    ],
}

STATUS_OPTIONS = ["delivered", "shipped", "processing", "cancelled", "returned"]
STATUS_WEIGHTS = [0.65, 0.15, 0.07, 0.08, 0.05]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def date_to_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    city          TEXT    NOT NULL,
    signup_date   TEXT    NOT NULL,   -- ISO 8601 YYYY-MM-DD
    is_premium    INTEGER NOT NULL DEFAULT 0  -- 1 = premium member
);

CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    category_id   INTEGER NOT NULL REFERENCES categories(category_id),
    unit_price    REAL    NOT NULL,
    stock         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    TEXT    NOT NULL,   -- ISO 8601 YYYY-MM-DD
    status        TEXT    NOT NULL    -- delivered / shipped / processing / cancelled / returned
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id       INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL    NOT NULL   -- price at time of purchase (may differ from current)
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id     INTEGER PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    rating        INTEGER NOT NULL,  -- 1 to 5
    review_date   TEXT    NOT NULL
);
"""

SCHEMA_INFO = SCHEMA_SQL.strip()  # exposed in Observation.schema_info


# ---------------------------------------------------------------------------
# Database builder
# ---------------------------------------------------------------------------

def build_database(seed: int = SEED) -> sqlite3.Connection:
    """
    Create and return a fully seeded in-memory SQLite connection.

    Always call with the same seed to guarantee reproducible ground truth.
    """
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    # --- Create schema ---
    conn.executescript(SCHEMA_SQL)

    # --- Categories ---
    for i, cat in enumerate(CATEGORIES, start=1):
        conn.execute("INSERT INTO categories VALUES (?, ?)", (i, cat))

    # --- Products ---
    product_id = 1
    product_rows = []
    for cat_idx, (cat_name, names) in enumerate(PRODUCT_NAMES.items(), start=1):
        for pname in names:
            price = round(rng.uniform(149.0, 4999.0), 2)
            stock = rng.randint(0, 200)
            conn.execute(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
                (product_id, pname, cat_idx, price, stock),
            )
            product_rows.append(product_id)
            product_id += 1

    # --- Customers ---
    start_signup = date(2020, 1, 1)
    end_signup   = date(2024, 6, 30)
    customer_ids = []
    used_emails: set[str] = set()

    for cid in range(1, 501):
        first = rng.choice(FIRST_NAMES)
        last  = rng.choice(LAST_NAMES)
        name  = f"{first} {last}"
        # make email unique
        base_email = f"{first.lower()}.{last.lower()}{cid}@example.com"
        email = base_email
        while email in used_emails:
            email = f"{first.lower()}.{last.lower()}{cid}_{rng.randint(1,999)}@example.com"
        used_emails.add(email)

        city        = rng.choice(CITIES)
        signup_date = date_to_str(random_date(rng, start_signup, end_signup))
        is_premium  = 1 if rng.random() < 0.25 else 0

        conn.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)",
            (cid, name, email, city, signup_date, is_premium),
        )
        customer_ids.append(cid)

    # --- Orders ---
    order_start = date(2022, 1, 1)
    order_end   = date(2024, 9, 30)
    order_id    = 1
    order_ids   = []

    for _ in range(1200):
        cid        = rng.choice(customer_ids)
        order_date = date_to_str(random_date(rng, order_start, order_end))
        status     = rng.choices(STATUS_OPTIONS, weights=STATUS_WEIGHTS, k=1)[0]

        conn.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?)",
            (order_id, cid, order_date, status),
        )
        order_ids.append(order_id)
        order_id += 1

    # --- Order Items ---
    item_id = 1
    for oid in order_ids:
        n_items = rng.randint(1, 4)
        chosen_products = rng.sample(product_rows, n_items)
        for pid in chosen_products:
            qty   = rng.randint(1, 3)
            # small price variance to simulate historical pricing
            base_price = conn.execute(
                "SELECT unit_price FROM products WHERE product_id = ?", (pid,)
            ).fetchone()[0]
            price = round(base_price * rng.uniform(0.9, 1.1), 2)
            conn.execute(
                "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)",
                (item_id, oid, pid, qty, price),
            )
            item_id += 1

    # --- Reviews ---
    review_id = 1
    review_start = date(2022, 3, 1)
    review_end   = date(2024, 9, 30)

    for _ in range(800):
        pid     = rng.choice(product_rows)
        cid     = rng.choice(customer_ids)
        rating  = rng.choices([1, 2, 3, 4, 5], weights=[0.05, 0.08, 0.15, 0.35, 0.37], k=1)[0]
        rdate   = date_to_str(random_date(rng, review_start, review_end))
        conn.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?)",
            (review_id, pid, cid, rating, rdate),
        )
        review_id += 1

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_sample_rows(conn: sqlite3.Connection, n: int = 3) -> dict[str, list[dict]]:
    """Return n sample rows from every table for inclusion in Observation."""
    tables = ["customers", "categories", "products", "orders", "order_items", "reviews"]
    samples = {}
    for table in tables:
        rows = conn.execute(f"SELECT * FROM {table} LIMIT {n}").fetchall()
        samples[table] = [dict(r) for r in rows]
    return samples


def execute_query(
    conn: sqlite3.Connection, query: str, row_limit: int = 50
) -> tuple[Optional[list[dict]], Optional[str]]:
    """
    Safely execute a SELECT query.

    Returns (results, None) on success or (None, error_message) on failure.
    Only SELECT statements are allowed — anything else raises an error.
    """
    stripped = query.strip().upper()

    # Safety: reject write operations
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "ATTACH", "PRAGMA")
    for kw in forbidden:
        if stripped.startswith(kw) or f" {kw} " in stripped:
            return None, f"Forbidden operation: {kw}. Only SELECT statements are allowed."

    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        return None, "Only SELECT (or WITH...SELECT) statements are permitted."

    try:
        cursor = conn.execute(query)
        rows   = cursor.fetchmany(row_limit)
        return [dict(r) for r in rows], None
    except sqlite3.Error as e:
        return None, str(e)


def row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]