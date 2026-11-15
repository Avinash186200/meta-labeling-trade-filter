"""CLI helper for account management. Usage:
    python manage.py create-account client_gc_live
"""
import sys
import storage

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "create-account":
        print("Usage: python manage.py create-account <account_id>")
        sys.exit(1)
    storage.init_db()
    account_id = sys.argv[2]
    api_key = storage.create_account(account_id)
    print(f"Account created: {account_id}")
    print(f"API key (give this to the client, store it securely -- shown once): {api_key}")
