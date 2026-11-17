"""Tests for the auth and user management API endpoints."""

from fastapi.testclient import TestClient

from backend.app.api.main import create_app


def test_auth_flow_check_register_login() -> None:
    app = create_app()
    client = TestClient(app)

    test_contact = f"user_{id(client)}@example.com"

    # 1. Check user initially does not exist
    resp = client.post("/api/check-user", json={"contact": test_contact})
    assert resp.status_code == 200
    data = resp.json()
    assert data["exists"] is False
    assert data["contact"] == test_contact

    # 2. Register user
    reg_payload = {
        "contact": test_contact,
        "password": "secretpassword123",
        "full_name": "Test User",
        "state": "Delhi",
        "district": "Central Delhi",
        "residence_area": "Urban",
        "age_group": "26-35",
        "preferred_language": "English",
        "highest_education": "Post Graduate",
        "education_status": "Completed",
        "primary_occupation": "Private Sector",
        "annual_income": "5-10 Lakhs",
        "is_active": True,
    }
    resp = client.post("/api/register", json=reg_payload)
    assert resp.status_code == 200
    reg_data = resp.json()
    assert reg_data["success"] is True
    assert "token" in reg_data

    # 3. Check user now exists
    resp = client.post("/api/check-user", json={"contact": test_contact})
    assert resp.status_code == 200
    assert resp.json()["exists"] is True

    # 4. Duplicate registration fails with 409
    resp = client.post("/api/register", json=reg_payload)
    assert resp.status_code == 409

    # 5. Login with invalid password fails
    resp = client.post(
        "/api/login",
        json={"contact": test_contact, "password": "wrongpassword"},
    )
    assert resp.status_code == 200
    login_fail = resp.json()
    assert login_fail["success"] is False
    assert "Incorrect password" in login_fail["error"]

    # 6. Login with correct password succeeds
    resp = client.post(
        "/api/login",
        json={"contact": test_contact, "password": "secretpassword123"},
    )
    assert resp.status_code == 200
    login_ok = resp.json()
    assert login_ok["success"] is True
    assert login_ok["user"]["full_name"] == "Test User"
    assert login_ok["user"]["contact"] == test_contact
    assert "token" in login_ok
