from fastapi.testclient import TestClient

from bdi_api.app import app

client = TestClient(app)


def test_list_aircraft():
    response = client.get("/api/s8/aircraft/?num_results=5&page=0")
    assert response.status_code == 200
    aircraft_list = response.json()
    # Since the DB might be empty, let’s just check the schema if there’s data
    if aircraft_list:
        for aircraft in aircraft_list:
            assert "icao" in aircraft
            assert isinstance(aircraft["icao"], str)
            # These fields are optional, so they might cdbe None
            assert "registration" in aircraft
            assert "type" in aircraft
            assert "owner" in aircraft
            assert "manufacturer" in aircraft
            assert "model" in aircraft

# Alright, let’s test the /aircraft/{icao}/co2 endpoint
def test_get_aircraft_co2():
    # Testing with a sample icao and day
    response = client.get("/api/s8/aircraft/a835af/co2?day=2023-11-01")
    assert response.status_code == 200
    co2_data = response.json()
    assert "icao" in co2_data
    assert co2_data["icao"] == "a835af"
    assert "hours_flown" in co2_data
    assert isinstance(co2_data["hours_flown"], float)
    assert "co2" in co2_data
    # co2 might be null if fuel rate isn’t found, so we just check the key exists

# co2 endpoint with an invalid date to make sure it fails properly
def test_get_aircraft_co2_invalid_date():
    response = client.get("/api/s8/aircraft/a835af/co2?day=invalid-date")
    assert response.status_code == 422  # FastAPI auto-validates and returns 422 for bad input
    assert "Invalid date format" in response.text
